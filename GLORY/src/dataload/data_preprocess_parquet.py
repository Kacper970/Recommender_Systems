import collections
import os
from pathlib import Path
from nltk.tokenize import word_tokenize
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import torch.nn.functional as F
from tqdm import tqdm
import random
import pickle
from collections import Counter
import numpy as np
import torch
import json
import itertools
import pandas as pd


def update_dict(target_dict, key, value=None):
    """
    Function for updating dict with key / key+value

    Args:
        target_dict(dict): target dict
        key(string): target key
        value(Any, optional): if None, equals len(dict+1)
    """
    if key not in target_dict:
        if value is None:
            target_dict[key] = len(target_dict) + 1
        else:
            target_dict[key] = value


def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)

# Rewrote this function to adapt it to new dataset format
def prepare_distributed_data(cfg, mode="train"):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir}
    target_file = os.path.join(data_dir[mode], f"behaviors_np{cfg.npratio}_0.tsv")
    if os.path.exists(target_file) and not cfg.reprocess:
        return 0
    print(f'Target file does not exist. Creating new behavior file at {target_file}')

    behaviors = []
    behavior_file_path = os.path.join(data_dir[mode], 'behaviors.parquet')
    df = pd.read_parquet(behavior_file_path)
    
    history_file_path = os.path.join(data_dir[mode], 'history.parquet')
    history_df = pd.read_parquet(history_file_path)

    df['user_id'] = df['user_id'].astype(str).str.strip()
    history_df['user_id'] = history_df['user_id'].astype(str).str.strip()

    user_history_dict = history_df.set_index('user_id')['article_id_fixed'].to_dict()
    
    all_articles = set(df['article_ids_inview'].explode().unique())

    if mode == 'train':
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            iid = str(row['impression_id'])
            uid = str(row['user_id'])
            time = str(row['impression_time'])
            
            if uid in user_history_dict:
                history = user_history_dict[uid]
            else:
                history = []

            clicked_articles = row['article_ids_clicked']

            neg = list(all_articles - set(history))
            pos = clicked_articles

            if len(pos) == 0 or len(neg) == 0:
                continue
            for pos_id in pos:
                neg_candidate = get_sample(neg, cfg.npratio)
                neg_str = ' '.join([str(x) for x in neg_candidate])
                history_str = ' '.join([str(x) for x in history])
                new_line = '\t'.join([iid, uid, time, history_str, str(pos_id), neg_str]) + '\n'
                behaviors.append(new_line)
        random.shuffle(behaviors)

        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        for i, line in enumerate(behaviors):
            behaviors_per_file[i % cfg.gpu_num].append(line)

    elif mode in ['val']:
        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            iid = str(row['impression_id'])
            uid = str(row['user_id'])
            time = str(row['impression_time'])
            
            if uid in user_history_dict:
                history = user_history_dict[uid]
            else:
                history = []

            inview_articles = row['article_ids_inview']
            clicked_articles = row['article_ids_clicked']
            
            # Prepare impression list with clicked and not clicked articles
            imp_list = []
            for article in inview_articles:
                if article in clicked_articles:
                    imp_list.append(f"{article}-1")
                else:
                    imp_list.append(f"{article}-0")
            
            imp = ' '.join(imp_list)
            history_str = ' '.join([str(x) for x in history])
            new_line = '\t'.join([iid, uid, time, history_str, imp]) + '\n'
            behaviors_per_file[idx % cfg.gpu_num].append(new_line)

    print(f'[{mode}] Writing files...')
    for i in range(cfg.gpu_num):
        processed_file_path = os.path.join(data_dir[mode], f'behaviors_np{cfg.npratio}_{i}.tsv')
        with open(processed_file_path, 'w') as f:
            f.writelines(behaviors_per_file[i])

    return len(behaviors)


def read_raw_news(cfg, file_path, mode='train'):
    """
    Function for reading the raw news file, articles.parquet

    Args:
        cfg:
        file_path(Path):                path of articles.parquet
        mode(string, optional):        train or val


    Returns:
        tuple:     (news, news_index, category_dict, subcategory_dict, word_dict)

    """
    import nltk
    nltk.download('punkt')
    
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir}

    if mode in ['val']:
        news_dict = pickle.load(open(Path(data_dir["train"]) / "news_dict.bin", "rb"))
        entity_dict = pickle.load(open(Path(data_dir["train"]) / "entity_dict.bin", "rb"))
        news = pickle.load(open(Path(data_dir["train"]) / "nltk_news.bin", "rb"))
    else:
        news = {}
        news_dict = {}
        entity_dict = {}

    category_dict = {}
    subcategory_dict = {}
    word_cnt = Counter()

    file_path = './data/ebnerd_small/articles.parquet'
    df = pd.read_parquet(file_path)

    entity_json_path = '../entity_emb/ebnerd_small.json'
    with open(entity_json_path, 'r') as file:
        entity_json = json.load(file)

    num_line = len(df)
    
    for idx, row in tqdm(df.iterrows(), total=num_line, desc=f"[{mode}] Processing raw news"):
        # Extracting relevant fields
        news_id = row['article_id']
        category = row['category']
        subcategory = str(row['subcategory'])
        title = row['title']
        entities = [e.lower() for e in row['ner_clusters']]
        if entities is None:
            entities = []
        
        update_dict(target_dict=news_dict, key=news_id)

        # Entity
        entity_ids = [entity_json["embeddings"][obj]["entity"] for obj in entities if obj in entity_json["embeddings"]]
        [update_dict(target_dict=entity_dict, key=entity_id) for entity_id in entity_ids]
        
        tokens = word_tokenize(title.lower(), language=cfg.dataset.dataset_lang)

        update_dict(target_dict=news, key=news_id, value=[tokens, category, subcategory, entity_ids,
                                                          news_dict[news_id]])

        if mode == 'train':
            update_dict(target_dict=category_dict, key=category)
            update_dict(target_dict=subcategory_dict, key=subcategory)
            word_cnt.update(tokens)

    if mode == 'train':
        word = [k for k, v in word_cnt.items() if v > cfg.model.word_filter_num]
        word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
        return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict
    else:  # val
        return news, news_dict, None, None, entity_dict, None


def read_parsed_news(cfg, news, news_dict,
                     category_dict=None, subcategory_dict=None, entity_dict=None,
                     word_dict=None):
    news_num = len(news) + 1
    news_category, news_subcategory, news_index = [np.zeros((news_num, 1), dtype='int32') for _ in range(3)]
    news_entity = np.zeros((news_num, 5), dtype='int32')

    news_title = np.zeros((news_num, cfg.model.title_size), dtype='int32')

    for _news_id in tqdm(news, total=len(news), desc="Processing parsed news"):
        _title, _category, _subcategory, _entity_ids, _news_index = news[_news_id]

        news_category[_news_index, 0] = category_dict[_category] if _category in category_dict else 0
        news_subcategory[_news_index, 0] = subcategory_dict[_subcategory] if _subcategory in subcategory_dict else 0
        news_index[_news_index, 0] = news_dict[_news_id]

        # entity
        entity_index = [entity_dict[entity_id] if entity_id in entity_dict else 0 for entity_id in _entity_ids]
        news_entity[_news_index, :min(cfg.model.entity_size, len(_entity_ids))] = entity_index[:cfg.model.entity_size]

        for _word_id in range(min(cfg.model.title_size, len(_title))):
            if _title[_word_id] in word_dict:
                news_title[_news_index, _word_id] = word_dict[_title[_word_id]]

    return news_title, news_entity, news_category, news_subcategory, news_index


def prepare_preprocess_bin(cfg, mode):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir}

    if cfg.reprocess is True:
        # Glove
        nltk_news, nltk_news_dict, category_dict, subcategory_dict, entity_dict, word_dict = read_raw_news(
            file_path=Path(data_dir[mode]) / "articles.parquet",
            cfg=cfg,
            mode=mode,
        )

        if mode == "train":
            pickle.dump(category_dict, open(Path(data_dir[mode]) / "category_dict.bin", "wb"))
            pickle.dump(subcategory_dict, open(Path(data_dir[mode]) / "subcategory_dict.bin", "wb"))
            pickle.dump(word_dict, open(Path(data_dir[mode]) / "word_dict.bin", "wb"))
        else:
            category_dict = pickle.load(open(Path(data_dir["train"]) / "category_dict.bin", "rb"))
            subcategory_dict = pickle.load(open(Path(data_dir["train"]) / "subcategory_dict.bin", "rb"))
            word_dict = pickle.load(open(Path(data_dir["train"]) / "word_dict.bin", "rb"))

        pickle.dump(entity_dict, open(Path(data_dir[mode]) / "entity_dict.bin", "wb"))
        pickle.dump(nltk_news, open(Path(data_dir[mode]) / "nltk_news.bin", "wb"))
        pickle.dump(nltk_news_dict, open(Path(data_dir[mode]) / "news_dict.bin", "wb"))
        nltk_news_features = read_parsed_news(cfg, nltk_news, nltk_news_dict,
                                              category_dict, subcategory_dict, entity_dict,
                                              word_dict)
        news_input = np.concatenate([x for x in nltk_news_features], axis=1)
        pickle.dump(news_input, open(Path(data_dir[mode]) / "nltk_token_news.bin", "wb"))
        print("Glove token preprocess finish.")
    else:
        print(f'[{mode}] All preprocessed files exist.')


def prepare_news_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir}

    nltk_target_path = Path(data_dir[mode]) / "nltk_news_graph.pt"

    reprocess_flag = False
    if not nltk_target_path.exists():
        reprocess_flag = True
        
    if not reprocess_flag and not cfg.reprocess:
        print(f"[{mode}] All graphs exist !")
        return
    
    # -----------------------------------------News Graph------------------------------------------------
    behavior_path = Path(data_dir['train']) / "behaviors.parquet"
    origin_graph_path = Path(data_dir['train']) / "nltk_news_graph.pt"

    news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    nltk_token_news = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    
    # ------------------- Build Graph -------------------------------
    if mode == 'train':
        edge_list, user_set = [], set()
        df = pd.read_parquet(behavior_path)
        num_line = len(df)
        
        for idx, row in tqdm(df.iterrows(), total=num_line, desc=f"[{mode}] Processing behaviors news to News Graph"):
            used_id = row['user_id']
            if used_id in user_set:
                continue
            else:
                user_set.add(used_id)
            
            history = row['article_ids_inview']
            long_edge = [news_dict[int(article)] for article in history if int(article) in news_dict]

            if len(long_edge) > 1:
                edge_list.append(long_edge)

        # edge count
        node_feat = nltk_token_news
        target_path = nltk_target_path
        num_nodes = len(news_dict) + 1

        short_edges = []
        for edge in tqdm(edge_list, total=len(edge_list), desc=f"Processing news edge list"):
            # Trajectory Graph
            if cfg.model.use_graph_type == 0:
                for i in range(len(edge) - 1):
                    short_edges.append((edge[i], edge[i + 1]))
            elif cfg.model.use_graph_type == 1:
                # Co-occurrence Graph
                for i in range(len(edge) - 1):
                    for j in range(i + 1, len(edge)):
                        short_edges.append((edge[i], edge[j]))
                        short_edges.append((edge[j], edge[i]))
            else:
                assert False, "Wrong"

        edge_weights = Counter(short_edges)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

        data = Data(x=torch.from_numpy(node_feat),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=num_nodes)
    
        torch.save(data, target_path)
        print(data)
        print(f"[{mode}] Finish News Graph Construction, \nGraph Path: {target_path} \nGraph Info: {data}")
    
    elif mode in ['val']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr
        node_feat = nltk_token_news

        data = Data(x=torch.from_numpy(node_feat),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(news_dict) + 1)
        
        torch.save(data, nltk_target_path)
        print(f"[{mode}] Finish nltk News Graph Construction, \nGraph Path: {nltk_target_path}\nGraph Info: {data}")

def prepare_neighbor_list(cfg, mode='train', target='news'):
    #--------------------------------Neighbors List-------------------------------------------
    print(f"[{mode}] Start to process neighbors list")

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir}

    neighbor_dict_path = Path(data_dir[mode]) / f"{target}_neighbor_dict.bin"
    weights_dict_path = Path(data_dir[mode]) / f"{target}_weights_dict.bin"

    reprocess_flag = False
    for file_path in [neighbor_dict_path, weights_dict_path]:
        if file_path.exists() is False:
            reprocess_flag = True
        
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] All {target} Neighbor dict exist !")
        return

    if target == 'news':
        target_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    elif target == 'entity':
        target_graph_path = Path(data_dir[mode]) / "entity_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    else:
        assert False, f"[{mode}] Wrong target {target} "

    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr

    if cfg.model.directed is False:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    neighbor_dict = collections.defaultdict(list)
    neighbor_weights_dict = collections.defaultdict(list)
    
    # for each node (except 0)
    for i in range(1, len(target_dict)+1):
        dst_edges = torch.where(edge_index[1] == i)[0]          # i as dst
        neighbor_weights = edge_attr[dst_edges]
        neighbor_nodes = edge_index[0][dst_edges]               # neighbors as src
        sorted_weights, indices = torch.sort(neighbor_weights, descending=True)
        neighbor_dict[i] = neighbor_nodes[indices].tolist()
        neighbor_weights_dict[i] = sorted_weights.tolist()
    
    pickle.dump(neighbor_dict, open(neighbor_dict_path, "wb"))
    pickle.dump(neighbor_weights_dict, open(weights_dict_path, "wb"))
    print(f"[{mode}] Finish {target} Neighbor dict \nDict Path: {neighbor_dict_path}, \nWeight Dict: {weights_dict_path}")


def prepare_entity_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir}

    target_path = Path(data_dir[mode]) / "entity_graph.pt"
    reprocess_flag = False
    if target_path.exists() is False:
        reprocess_flag = True
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] Entity graph exists!")
        return

    entity_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
    origin_graph_path = Path(data_dir['train']) / "entity_graph.pt"

    if mode == 'train':
        target_news_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        news_graph = torch.load(target_news_graph_path)
        print("news_graph,", news_graph)
        entity_indices = news_graph.x[:, -8:-3].numpy()
        print("entity_indices, ", entity_indices.shape)

        entity_edge_index = []
        # -------- Inter-news -----------------

        news_edge_src, news_edge_dest = news_graph.edge_index
        edge_weights = news_graph.edge_attr.long().tolist()
        for i in range(news_edge_src.shape[0]):
            src_entities = entity_indices[news_edge_src[i]]
            dest_entities = entity_indices[news_edge_dest[i]]
            src_entities_mask = src_entities > 0
            dest_entities_mask = dest_entities > 0
            src_entities = src_entities[src_entities_mask]
            dest_entities = dest_entities[dest_entities_mask]
            edges = list(itertools.product(src_entities, dest_entities)) * edge_weights[i]
            entity_edge_index.extend(edges)

        edge_weights = Counter(entity_edge_index)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

        # --- Entity Graph Undirected
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        data = Data(x=torch.arange(len(entity_dict) + 1),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(entity_dict) + 1)
            
        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")
    elif mode in ['val']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr

        data = Data(x=torch.arange(len(entity_dict) + 1),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(entity_dict) + 1)
        
        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")

# Removed any function calls relating to test subset
def prepare_preprocessed_data(cfg):
    prepare_distributed_data(cfg, "train")
    prepare_distributed_data(cfg, "val")

    prepare_preprocess_bin(cfg, "train")
    prepare_preprocess_bin(cfg, "val")

    prepare_news_graph(cfg, 'train')
    prepare_news_graph(cfg, 'val')

    prepare_neighbor_list(cfg, 'train', 'news')
    prepare_neighbor_list(cfg, 'val', 'news')

    prepare_entity_graph(cfg, 'train')
    prepare_entity_graph(cfg, 'val')

    prepare_neighbor_list(cfg, 'train', 'entity')
    prepare_neighbor_list(cfg, 'val', 'entity')

    # Entity vec process
    data_dir = {"train":cfg.dataset.train_dir, "val":cfg.dataset.val_dir}
    train_entity_emb_path = Path(data_dir['train']) / "entity_embedding.vec"
    val_entity_emb_path = Path(data_dir['val']) / "entity_embedding.vec"

    val_combined_path = Path(data_dir['val']) / "combined_entity_embedding.vec"

    os.system("cat " + f"{train_entity_emb_path} {val_entity_emb_path}" + f" > {val_combined_path}")

