import json
import os
from tqdm import tqdm
from typing import List, Dict, Tuple
import re
import argparse
import numpy as np


def load_jsonl(path):
    with open(path) as fp:
        return [json.loads(line) for line in fp.readlines()]


def stringify_example(context: Dict, selected_steps: List, hypothesis: str, randomize_ids=True) -> Tuple[str, str]:
    # shuffle dict of contet
    context_str = ""
    if randomize_ids:
        keys = list(context.keys())
        np.random.shuffle(keys)
        context = [(k, context[k]) for k in keys]
        for key, value in context:
            context_str += f'{key}: {value}\n'

    np.random.shuffle(selected_steps)
    selected_steps_str = " & ".join(selected_steps)
    return f"premises:\n{context_str}hypothesis:\n{hypothesis}\nproof steps: ", selected_steps_str


def extract_ruletaker_steps(example: Dict):
    results = []

    sents = example['context']
    sents = re.split(r'sent\d+:', sents)
    sents = [s.strip() for s in sents if len(s) > 0]
    sents_d = {}
    for i, s in enumerate(sents):
        sents_d[f'sent{i+1}'] = s

    #todo: check if number of proofs for examples
    proofs = example['proofs'][0]
    proofs = [p.strip() for p in proofs.split(';') if len(p) > 0]

    for proof in proofs:
        premises, conclusion = proof.split('->')
        premises = [p.strip() for p in premises.split('&')]
        conclusion = conclusion.strip()

        results.append(stringify_example(sents_d, premises, example['hypothesis'], randomize_ids=True))

        match = re.findall(r'^(int\d+):', conclusion)
        if match:
            match = match[0].strip()
            conclusion = re.sub(r'^int\d+:', '', conclusion).strip()
            sents_d[match] = conclusion

    return results


def preprocess_ruletaker(base_path, split='dev'):

    extracted_steps = []
    for depth in [1, 2, 3, 5]:
        print("extracting depth", depth, '...')

        dataset = load_jsonl(os.path.join(base_path, f'depth-{depth}', f'meta-{split}.jsonl'))
        for example in tqdm(dataset):
            if example['depth'] is None or example['depth'] == 0 or example['answer'] is False:
                continue
            extracted_steps.extend(extract_ruletaker_steps(example))

    return extracted_steps


def extract_entailmenttree_steps(example: Dict):

    results = []
    sents = example['context']
    sents = re.split(r'sent\d+:', sents)
    sents = [s.strip() for s in sents if len(s) > 0]
    sents_d = {}
    for i, s in enumerate(sents):
        sents_d[f'sent{i+1}'] = s

    #todo: check if number of proofs for examples
    proofs = example['proof']
    proofs = [p.strip() for p in proofs.split(';')][:-1]

    for proof in proofs:
        premises, conclusion = proof.split('->')
        premises = [p.strip() for p in premises.split('&')]
        conclusion = conclusion.strip()

        results.append(stringify_example(sents_d, premises, example['hypothesis'], randomize_ids=True))

        match = re.findall(r'^(int\d+):', conclusion)
        if match:
            match = match[0].strip()
            conclusion = re.sub(r'^int\d+:', '', conclusion).strip()
            sents_d[match] = conclusion
        elif conclusion == 'hypothesis':
            continue
        else:
            raise Exception("Invalid conclusion")

    return results

def preprocess_entailmenttree(base_path, split='dev'):

    extracted_steps = []
    for task in ['task_1', 'task_2']:
        print("extracting from ", task, '...')

        dataset = load_jsonl(os.path.join(base_path, f'dataset/{task}', f'{split}.jsonl'))
        for example in tqdm(dataset):
            if example['depth_of_proof'] is None or example['depth_of_proof'] == 0:
                continue
            extracted_steps.extend(extract_entailmenttree_steps(example))

    return extracted_steps


def merge_entailmenttree_ruletaker(paths, output, split='train', merge_equal=False):
    all_lines = []
    ds_list = []
    for path in paths:
        print("loading from ", path, "...")
        train_path = os.path.join(path, f'{split}.jsonl')
        with open(train_path, 'r') as fp:
            lines = fp.readlines()
            ds_list.append(lines)

    if merge_equal:
        sizes = [len(ds) for ds in ds_list]
        min_size = min(sizes)

        for i, ds in enumerate(ds_list):
            np.random.shuffle(ds)
            all_lines.extend(ds[:min_size])

    with open(output, 'w') as fp:
        np.random.shuffle(all_lines)
        fp.writelines(all_lines)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_path', type=str, default='data/proofwriter-dataset-V2020.12.3/preprocessed_OWA')
    # parser.add_argument('--output_path', type=str, default='data/proofwriter-selector/')
    parser.add_argument('--dataset', type=str, default='entailmenttree', choices=['entailmenttree', 'ruletaker'])
    parser.add_argument('--merge', nargs='+', default=None, help='a list of preprocessed dataset to merge')
    parser.add_argument('--input_path', type=str, default='data/entailment_trees_emnlp2021_data_v3')
    parser.add_argument('--output_path', type=str, default='data/entailmenttree-selector/')
    # parser.add_argument('--output_path', type=str, default='data/selector_dev_merged.json')
    parser.add_argument('--merge-equal', action='store_true', help='merge equal number of examples from each dataset')
    args = parser.parse_args()

    if args.merge is not None:
        split = 'train' if 'train' in args.output_path else 'dev'
        merge_entailmenttree_ruletaker(args.merge, args.output_path, split=split, merge_equal=args.merge_equal)
        return

    os.makedirs(args.output_path, exist_ok=True)
    for split in ['dev', 'test', 'train']:
        if args.dataset == 'entailmenttree':
            extracted_examples = preprocess_entailmenttree(args.input_path, split)
        elif args.dataset == 'ruletaker':
            extracted_examples = preprocess_ruletaker(args.input_path, split)
        else:
            raise NotImplementedError()
        if split == 'train':
            np.random.shuffle(extracted_examples)

        print("writing to file...")
        with open(os.path.join(args.output_path, f'{split}.jsonl'), 'w') as fp:
            for ex in extracted_examples:
                fp.write(json.dumps({'premises': ex[0], 'steps': ex[1]}) + '\n')


if __name__ == '__main__':
    main()
