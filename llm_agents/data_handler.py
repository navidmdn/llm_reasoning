import json
import numpy as np
import re

def load_dataset(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    return data


def get_processed_entailmenet_dataset(train_data, valid_data, identifier_description=False):
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    def process_context(triples):
        return '\n'.join([f"{k}: {v}" for k, v in triples.items()])

    if identifier_description:
        for ex in train_data:
            raw_proof = ex['proof']
            ids = re.findall(r'(sent\d+)', raw_proof)
            for id in ids:
                raw_proof = raw_proof.replace(id, f"{id}: {ex['meta']['triples'][id]}")
            ex['proof'] = raw_proof

    return (
        [{'hypothesis': s['hypothesis'], 'context': process_context(s['meta']['triples']),
          'proof': s['proof']} for s in train_data],
        [{'hypothesis': s['hypothesis'], 'context': process_context(s['meta']['triples']),
          'proof': s['proof']} for s in valid_data])

def get_processed_proofwriter_dataset(train_data, valid_data):
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)
    train = []
    valid = []
    for ex in train_data:
        if len(ex['proofs']) > 0:
            train.append({'hypothesis': ex['hypothesis'], 'context': ex['context'], 'proof': ex['proofs'][0]})

    for ex in valid_data:
        if len(ex['proofs']) > 0:
            valid.append({'hypothesis': ex['hypothesis'], 'context': ex['context'], 'proof': ex['proofs'][0]})
    return train, valid


