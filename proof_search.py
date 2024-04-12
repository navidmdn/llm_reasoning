from fire import Fire
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import json
from typing import List, Dict, Tuple, Set
import re

class SearchState:
    def __init__(self, example: Dict):
        if 'meta' in example:
            self.dataset = 'entailmentbank'
        else:
            self.dataset = 'ruletaker'

        self.preprocess_example(example, self.dataset)
        self.intermediates = {}
        self.proof_steps = []

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
        #todo: a sort of check between nodes and hypothesis to see if we've reached the hypothesis
        return False

    def add_proof_step(self, proof: str, step: Set[str]):
        next_int_id = self.num_inferred_nodes + 1
        self.intermediates[f'int{next_int_id}'] = proof
        self.context[f'int{next_int_id}'] = proof
        step_txt = " & ".join(step)
        self.proof_steps.append(f"{step_txt} -> int{next_int_id}: {proof}")


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
        if self.dataset == 'entailmentbank':
            self.preprocess_entailmentbank_example(example)
        elif self.dataset == 'ruletaker':
            self.preprocess_ruletaker_example(example)
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")


def process_generated_steps(generated_steps: str) -> Set[str]:
    if " & " not in generated_steps:
        return set([])

    steps = generated_steps.split(" & ")
    return set([step.strip() for step in steps])

def sample_steps(selector: PreTrainedModel, selector_tokenizer: PreTrainedTokenizer, search_state: SearchState,
                 top_k: int = 3) ->\
        Tuple[List[Set[str]], List[float]]:
    """
    Sample the next steps to take in the proof search. It will return a list which each element is a list of
    identifiers of the steps to take. Also returns a list of scores for each of the sampled steps.
    """
    prefix = "select the best steps for induction in forward reasoning from the following premises:\n"
    input_txt = f"{prefix}{search_state.get_context_str()}"

    inputs = selector_tokenizer(input_txt, return_tensors='pt')
    print(inputs['input_ids'].shape)

    outputs = selector.generate(**inputs, max_length=100, num_return_sequences=top_k, num_beams=top_k,
                                do_sample=True)
    step_texts = selector_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    samples = set()

    for step_text in step_texts:
        step_ids = process_generated_steps(step_text)
        if len(step_ids) == 0:
            continue
        samples.add(step_ids)

    assert len(samples) > 0, "No valid steps were sampled"
    #todo: calculate scores for each of the sampled steps
    return list(samples), [1.0] * len(samples)


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
    inputs = deductor_tokenizer(input_txt, return_tensors='pt')
    outputs = deductor.generate(**inputs, max_length=100, num_return_sequences=1, num_beams=5)
    generated_text = deductor_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    search_state.add_proof_step(generated_text, next_step)
    return search_state

def greedy_proof_search(example: Dict, deductor: PreTrainedModel, deductor_tokenizer: PreTrainedTokenizer,
                 selector: PreTrainedModel, selector_tokenizer: PreTrainedTokenizer, n_iters: int = 3):
    search_state = SearchState(example)
    proof = ""
    iterations = 0

    while not search_state.has_reached_hypothesis() and iterations < n_iters:
        next_steps, _ = sample_steps(selector, selector_tokenizer, search_state)
        next_step = next_steps[0]
        search_state = apply_deductor(deductor, deductor_tokenizer, search_state, next_step)
        iterations += 1

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

    for example in examples:
        proof_search(example, deductor, deductor_tokenizer, selector, selector_tokenizer)


if __name__ == '__main__':
    Fire(run)