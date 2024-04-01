import json
import numpy as np


def load_entailemnt_tree_dataset(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    return data


def get_processed_entailmenet_dataset(train_data, valid_data):
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    def process_context(triples):
        return '\n'.join([f"{k}: {v}" for k, v in triples.items()])

    return (
        [{'hypothesis': s['hypothesis'], 'context': process_context(s['meta']['triples']),
          'proof': s['proof']} for s in train_data],
        [{'hypothesis': s['hypothesis'], 'context': process_context(s['meta']['triples']),
          'proof': s['proof']} for s in valid_data])


