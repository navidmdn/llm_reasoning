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


def get_processed_entailmenet_dataset_textual(train_data, valid_data):
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    def process_context(triples):
        return '\n'.join([f"{k}: {v}" for k, v in triples.items()])


    for ex in train_data:
        raw_proof = ex['full_text_proof']
        steps = raw_proof.split('[BECAUSE]')[1:]
        steps = [s.strip() for s in steps]

        state = ex['meta']['triples'].copy()
        inv_state = {v: k for k, v in state.items()}

        steps_updated = []
        for step in steps:
            prems, conc = step.split('[INFER]')
            prems = prems.strip()
            conc = conc.strip()
            prems = prems.split('[AND]')
            prems = [p.strip() for p in prems]

            prems_updated = []
            for p in prems:
                intermediate = re.findall(r'(int\d+)', p)
                if len(intermediate) > 0:
                    int_id = intermediate[0]
                    int_desc = p.replace(f"{int_id}", f"{int_id}: {state[int_id]}").strip()
                    prems_updated.append(int_desc)
                else:
                    prems_updated.append(f"{inv_state[p]}: {p}")

            intermediate = re.findall(r'(int\d+)', conc)
            if len(intermediate) > 0:
                int_id = intermediate[0]
                int_desc = conc.replace(f"{int_id}:", "").strip()
                state[int_id] = int_desc

            steps_updated.append(f"{' [AND] '.join(prems_updated)} [INFER] {conc}")
            ex['proof'] = f"[BECAUSE] {' [BECAUSE] '.join(steps_updated)}"

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


