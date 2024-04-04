import fire
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json


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
        inp_txt = f"infer: {example['premises']}"
        print("premises: ",inp_txt)

        inputs = tokenizer([inp_txt], return_tensors="pt")
        output = model.generate(**inputs, max_length=100, num_beams=4, no_repeat_ngram_size=2, num_return_sequences=1)
        output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        print("output: ", output)
        print("ground truth: ", example["conclusion"])
        input()


if __name__ == '__main__':
    fire.Fire(run)