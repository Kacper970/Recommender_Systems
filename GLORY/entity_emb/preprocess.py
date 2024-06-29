import pandas as pd
from collections import Counter
import json

articles_path = '../GLORY/data/ebnerd_small/articles.parquet'
df = pd.read_parquet(articles_path, engine='pyarrow')

print("Loaded parquet.")

entity_counter = Counter()
print(df.columns)
for x in df['ner_clusters']:
    entity_counter.update([word.lower() for word in x])

print(f"The dataset contains {len(entity_counter)} unique entities.")

json_path = 'ebnerd_small.json'
try:
    with open(json_path, 'r') as file:
        stats = json.load(file)
except:
    stats = {}
stats['most_common'] = list(entity_counter.most_common())

with open(json_path, 'w') as file:
    json.dump(stats, file, indent=2)