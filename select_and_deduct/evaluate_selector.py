import fire
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from typing import List, Dict, Tuple
import re


def has_correct_format(pred: str) -> bool:
    if ' & ' in pred:
        pred = pred.split(' & ')
        pred = [p.strip() for p in pred]

        for p in pred:
            if not re.match(r'^int\d+$', p) and not re.match(r'^sent\d+$', p):
                return False
        return True
    # this is the case that the model thinks the hypothesis is in the context:
    elif re.match(r'^int\d+$', pred.strip()) or re.match(r'^sent\d+$', pred.strip()):
        return True
    return False

def _permutation_invariant_match(pred: str, target: str) -> bool:
    if not has_correct_format(pred):
        return False

    pred = pred.split(' & ')
    target = target.split(' & ')

    if len(pred) != len(target):
        return False

    return set(pred) == set(target)


def permutation_invariant_metrics(preds: List[str], targets: List[str], num_pred_seq: int) -> Dict[str, float]:
    i = 0
    topk_match = 0
    top1_match = 0
    diversity_scores = []

    while i < len(targets):
        found_match = False
        distinct_preds = set()
        for k, j in enumerate(range(i*num_pred_seq, i*num_pred_seq+num_pred_seq)):
            match = _permutation_invariant_match(preds[j], targets[i])
            if match:
                found_match = True
                if k == 0:
                    top1_match += 1

            if has_correct_format(preds[j]):
                distinct_preds.add(preds[j])

        if found_match:
            topk_match += 1

        diversity_scores.append(len(distinct_preds) / num_pred_seq)
        i += 1

    return {
        'top1_acc': top1_match / len(targets),
        f'top{num_pred_seq}_acc': topk_match / len(targets),
        'diversity': np.mean(diversity_scores),
        'diversity_acc_score': np.mean(diversity_scores) * (topk_match / len(targets))
    }


def load_hf_model_and_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_examples(test_path):
    with open(test_path, 'r') as f:
        examples = f.readlines()
    return [json.loads(example) for example in examples]


def run(model_name, test_path):
    examples = load_examples(test_path)
    np.random.shuffle(examples)

    model, tokenizer = load_hf_model_and_tokenizer(model_name)

    for example in examples:
        inp_txt = f"select steps: {example['premises']}"
        print("premises: ", inp_txt)

        inputs = tokenizer([inp_txt], return_tensors="pt")
        output = model.generate(**inputs, max_length=100, num_beams=10, no_repeat_ngram_size=2, num_return_sequences=10)

        output = tokenizer.batch_decode(output, skip_special_tokens=True)
        for i, out in enumerate(output):
            print(f"output {i}: ", out)

        print("ground truth: ", example["steps"])
        input()


if __name__ == '__main__':
    fire.Fire(run)