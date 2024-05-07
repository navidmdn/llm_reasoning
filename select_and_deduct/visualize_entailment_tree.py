from treelib import Node, Tree
from fire import Fire
import json
import re


def visualize_tree(prediction: str, test_example: dict):
    tree = Tree()
    sents = test_example['context']
    sents = re.split(r'sent\d+:', sents)
    sents = [s.strip() for s in sents if len(s) > 0]
    context = {}
    for i, s in enumerate(sents):
        context[f'sent{i + 1}'] = s

    pred_steps = prediction.replace("$proof$ =", "").strip().split(";")[:-1]
    hypothesis = test_example["hypothesis"]

    for step in pred_steps:
        premises, conclusion = step.split("->")
        conclusion = conclusion.strip()

        if conclusion != 'hypothesis':
            conc_id = re.findall(r'^(int\d+):', conclusion)[0].strip()
            conclusion = re.sub(r'^int\d+:', '', conclusion).strip()
            context[conc_id] = conclusion

    for step in pred_steps[::-1]:
        premises, conclusion = step.split("->")
        premises = [p.strip() for p in premises.split("&")]
        conclusion = conclusion.strip()

        if conclusion == 'hypothesis':
            conc_id = "hypothesis"
            tree.create_node(hypothesis, "hypothesis")
        else:
            conc_id = re.findall(r'^(int\d+):', conclusion)[0].strip()

        for p in premises:
            if p in tree.nodes:
                continue
            tree.create_node(context[p], p, parent=conc_id)

    return tree.show(stdout=False)


def main(pred_file: str, test_file: str, output_file: str = None):
    with open(pred_file, 'r') as f:
        pred = f.readlines()

    test_examples = []
    with open(test_file, 'r') as f:
        for line in f:
            test_examples.append(json.loads(line))

    for p, e in zip(pred, test_examples):
        print(visualize_tree(p, e))
        input()


if __name__ == '__main__':
    Fire(main)