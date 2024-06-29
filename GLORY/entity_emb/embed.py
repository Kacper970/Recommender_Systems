import os
os.environ["PREFIX"] = ""

import pickle
import graphvite as gv
import json

with open("simple_wikidata5m.pkl", "rb") as fin:
    model = pickle.load(fin)
entity2id = model.graph.entity2id
relation2id = model.graph.relation2id
entity_embeddings = model.solver.entity_embeddings
relation_embeddings = model.solver.relation_embeddings

print("loaded pickle.")

alias2entity = gv.dataset.wikidata5m.alias2entity
# alias2relation = gv.dataset.wikidata5m.alias2relation
# print(entity_embeddings[entity2id[alias2entity["machine learning"]]])
# print(relation_embeddings[relation2id[alias2relation["field of work"]]])

json_path = 'ebnerd_small.json'
with open(json_path, 'r') as file:
    stats = json.load(file)

stats['embeddings'] = {}
for word, _ in stats['most_common']:
    if word in alias2entity:
        entity = alias2entity[word]
        if entity not in entity2id:
            print('could not find entity:', entity)
            continue
        id = entity2id[entity]
        vector = entity_embeddings[id]
        stats['embeddings'][word] = {
            'entity': entity,
            'id': id,
            'vector': vector.tolist()
        }

with open(json_path, 'w') as file:
    json.dump(stats, file, indent=2)