from datasets import load_dataset
import json
from fire import Fire
from typing import List, Dict
from tqdm import tqdm

INSTRUCTION_BEGIN_TOKENS = "### Instruction\n"
RESPONSE_BEGIN_TOKENS = "### Response\n"

def preprocess_deduction(data: List[Dict]) -> List[Dict]:
    system_message = """You are the entailment component of a reasoning system. Your task is to take a couple of sentences and induce\
 a new sentence."""

    result = []
    for sample in tqdm(data):
        text = f"{INSTRUCTION_BEGIN_TOKENS}{system_message}\n{sample['premises']}\n{RESPONSE_BEGIN_TOKENS}{sample['conclusion']}"
        result.append({'text': text})

    return result


def preprocess_selection(data: List[Dict]) -> List[Dict]:
    pass


def preprocess(data: List[Dict], task: str) -> List[Dict]:
    if task == "deduction":
        data = preprocess_deduction(data)
    elif task == "selection":
        data = preprocess_selection(data)
    else:
        raise ValueError(f"Task {task} not supported")
    return data


def run(input_file: str, output_file: str, task: str = "deduction"):
    data = []
    with open(input_file) as f:
        for line in f:
            data.append(json.loads(line))

    data = preprocess(data, task=task)

    with open(output_file, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    Fire(run)
