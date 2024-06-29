import json
import numpy as np

json_path = 'ebnerd_small.json'
with open(json_path, 'r') as file:
    stats = json.load(file)

def images():
    import matplotlib.pyplot as plt
    can_be_embedded = []
    for word, _ in stats['most_common']:
        can_be_embedded.append(word in stats['embeddings'])
    y = np.cumsum(can_be_embedded)
    x = np.arange(len(y))

    plt.plot(x, y)
    plt.savefig('most_common_embeddable.png')

def write_vec_embs():
    vec_path = '/home/scur1567/GLORY/data/ebnerd_small/train/entity_embedding.vec'
    lines = []
    for entity in stats['embeddings'].values():
        entity, _, vector = entity.values()
        lines.append(' '.join([entity] + [str(x) for x in vector]) + '\n')
    with open(vec_path, 'w') as file:
        file.writelines(lines)

def main():
    # images()
    write_vec_embs()

if __name__ == '__main__':
    main()