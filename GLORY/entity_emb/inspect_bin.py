import pickle
picke_path = '/home/scur1567/GLORY/data/ebnerd_small/train/entity_dict.bin'
# picke_path = '/home/scur1567/GLORY/data/MINDsmall/train/entity_dict.bin'
dict = pickle.load(open(picke_path, "rb"))
print(dict)