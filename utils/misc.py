import json


def load_prompt(prompt_path: str) -> str:
    assert prompt_path.endswith('.txt')
    with open(prompt_path, 'r') as f:
        prompt = f.read().strip()
    return prompt

def load_jsonl(file_path: str) -> list:
    assert file_path.endswith('.jsonl') or file_path.endswith('.json')
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data