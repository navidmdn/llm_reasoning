import numpy as np
from fire import Fire
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import json
from typing import List, Dict, Tuple, Set, Union, FrozenSet
import re
from tqdm import tqdm


class SearchState:
    def __init__(self, example: Dict):
        if 'meta' in example:
            self.dataset = 'entailmentbank'
        else:
            self.dataset = 'ruletaker'

        self.preprocess_example(example, self.dataset)
        self.intermediates = {}
        self.proof_steps = []

    def __repr__(self):
        current_context = "\n".join([f"{k}: {v}" for k, v in self.context.items()])
        intermediates = "\n".join([f"{k}: {v}" for k, v in self.intermediates.items()])
        return f"Current context:\n{current_context}\nIntermediates:\n{intermediates}\nHypothesis: {self.hypothesis}"

    @property
    def num_inferred_nodes(self):
        return len(self.intermediates)

    def preprocess_entailmentbank_example(self, example: Dict):
        self.context = self.process_identified_context(example['context'])
        self.hypothesis = example['hypothesis']

    def preprocess_ruletaker_example(self, example: Dict):
        self.context = self.process_identified_context(example['context'])
        self.hypothesis = example['hypothesis']

    def has_reached_hypothesis(self) -> bool:
        #todo: implement a better hypothesis for this (rouge1/f1/etc.)
        last_proof = self.get_last_proof()
        if last_proof is None:
            return False
        return self.hypothesis == last_proof

    def add_proof_step(self, proof: str, step: Set[str]):
        next_int_id = self.num_inferred_nodes + 1
        self.intermediates[f'int{next_int_id}'] = proof
        self.context[f'int{next_int_id}'] = proof
        step_txt = " & ".join(step)
        self.proof_steps.append(f"{step_txt} -> int{next_int_id}: {proof}")

    def get_selection_prompt(self) -> str:
        prefix = "select the best steps for induction in forward reasoning from the following premises:\n"
        premises = "\n".join([f"{k}: {v}" for k, v in self.context.items()])
        context = f"premises:\n{premises}\nhypothesis:\n{self.hypothesis}\nproof steps: "
        return f"{prefix}{context}"

    def get_last_proof(self) -> Union[str, None]:
        if len(self.intermediates) == 0:
            return None
        return self.intermediates[f'int{self.num_inferred_nodes}']

    def process_identified_context(self, context: str) -> Dict[str, str]:
        """
        Process the context of the proof to extract the sentences and their identifiers.
        it should be in the format of dataset we have like: "sent1: some sentence sent2: another sentence ..."
        """
        sents = re.split(r'sent\d+:', context)
        sents = [s.strip() for s in sents if len(s) > 0]
        sents_d = {}
        for i, s in enumerate(sents):
            sents_d[f'sent{i + 1}'] = s
        return sents_d

    def preprocess_example(self, example: Dict, dataset: str):
        if dataset == 'entailmentbank':
            self.preprocess_entailmentbank_example(example)
        elif dataset == 'ruletaker':
            self.preprocess_ruletaker_example(example)
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")


def process_generated_steps(generated_steps: str) -> FrozenSet[str]:
    if " & " not in generated_steps:
        return frozenset([])

    steps = generated_steps.split(" & ")
    return frozenset([step.strip() for step in steps])


def sample_steps(selector: PreTrainedModel, selector_tokenizer: PreTrainedTokenizer, search_state: SearchState,
                 top_k: int = 5) ->\
        Tuple[List[FrozenSet[str]], List[float]]:
    """
    Sample the next steps to take in the proof search. It will return a list which each element is a list of
    identifiers of the steps to take. Also returns a list of scores for each of the sampled steps.
    """
    input_txt = search_state.get_selection_prompt()

    # print("selection prompt:")
    # pprint(input_txt)

    inputs = selector_tokenizer(input_txt, return_tensors='pt')
    outputs = selector.generate(**inputs, max_length=100, num_return_sequences=top_k, num_beams=top_k,
                                do_sample=True, return_dict_in_generate=True, output_scores=True)

    step_texts = selector_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    scores = outputs.sequences_scores
    samples = []
    sample_scores = []

    for step_text, score in zip(step_texts, scores):
        step_ids = process_generated_steps(step_text)
        if len(step_ids) == 0:
            continue
        if step_ids not in samples:
            samples.append(step_ids)
            sample_scores.append(score)

    assert len(samples) > 0, "No valid steps were sampled"
    return samples, sample_scores


def apply_deductor(deductor, deductor_tokenizer, search_state: SearchState, next_step: Set[str]) -> SearchState:
    """
    Apply the deductor model to the current proof state and the next steps to take. It will return the updated search
    state with the newly created node.
    """
    prefix = "infere: "
    step_texts = []
    for step in next_step:
        step_texts.append(search_state.context[step].strip())

    input_txt = prefix + " [AND] ".join(step_texts) + " [INFER]"
    # print(f"deductor input: {input_txt}")
    inputs = deductor_tokenizer(input_txt, return_tensors='pt')
    outputs = deductor.generate(**inputs, max_length=100, num_return_sequences=1, num_beams=5)
    generated_text = deductor_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    search_state.add_proof_step(generated_text, next_step)
    return search_state


def greedy_proof_search(example: Dict, deductor: PreTrainedModel, deductor_tokenizer: PreTrainedTokenizer,
                 selector: PreTrainedModel, selector_tokenizer: PreTrainedTokenizer, n_iters: int = 6):
    search_state = SearchState(example)
    iterations = 0
    print(search_state)

    while not search_state.has_reached_hypothesis() and iterations < n_iters:
        # pprint(f"step {iterations + 1}")
        next_steps, _ = sample_steps(selector, selector_tokenizer, search_state)
        next_step = next_steps[0]
        # print("Next step: ", next_step)
        search_state = apply_deductor(deductor, deductor_tokenizer, search_state, next_step)
        # pprint("Current proof step: " + search_state.proof_steps[-1])
        iterations += 1

    for proof_step in search_state.proof_steps:
        pprint(proof_step)


def pprint(txt: str):
    print("====================================")
    print(txt)
    print("====================================")

def load_examples(test_data_path: str):
    with open(test_data_path, 'r') as f:
        examples = f.readlines()
    return [json.loads(example) for example in examples]

def load_model_and_tokenizer(model_name: str):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def run(deductor_path: str, selector_path: str, test_data_path: str, ):

    deductor, deductor_tokenizer = load_model_and_tokenizer(deductor_path)
    selector, selector_tokenizer = load_model_and_tokenizer(selector_path)

    examples = load_examples(test_data_path)
    np.random.shuffle(examples)

    for example in tqdm(examples):
        #todo: make sure false examples are not in dataset
        if 'depth' in example and (example['depth'] is None or example['depth'] < 1 or example['answer'] is False):
            continue
        greedy_proof_search(example, deductor, deductor_tokenizer, selector, selector_tokenizer)

        print("*" * 50)
        print("Ground truth proof:")
        print(example['proof'] if 'proof' in example else example['proofs'])
        print("*" * 50)

        input()

if __name__ == '__main__':
    Fire(run)
