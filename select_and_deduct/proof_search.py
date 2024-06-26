import os

import numpy as np
from fire import Fire
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import json
from typing import List, Dict, Tuple, Set, Union, FrozenSet
import re
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from copy import deepcopy
from collections import OrderedDict
from visualize_entailment_tree import visualize_tree


def print_gpu_utilization():
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


class SearchState:
    def __init__(self, example: Union[Dict, None], bluert_model: PreTrainedModel,
                 bluert_tokenizer: PreTrainedTokenizer):

        if example is None:
            self.dataset = None
        else:
            if 'meta' in example:
                self.dataset = 'entailmentbank'
            else:
                self.dataset = 'ruletaker'
            self.preprocess_example(example, self.dataset)

        self.intermediates = OrderedDict()
        self.proof_steps = []
        self.bleurt_model = bluert_model
        self.bleurt_tokenizer = bluert_tokenizer
        self.score = 0

    def __repr__(self):
        current_context = "\n".join([f"{k}: {v}" for k, v in self.context.items()])
        intermediates = "\n".join([f"{k}: {v}" for k, v in self.intermediates.items()])
        return f"Current context:\n{current_context}\nIntermediates:\n{intermediates}\nHypothesis: {self.hypothesis}"

    @property
    def num_inferred_nodes(self):
        return len(self.intermediates)

    def get_formatted_full_proof(self) -> str:
        if len(self.proof_steps) == 0:
            return "$proof$ = "
        last_step = self.proof_steps[-1]
        prev_steps = self.proof_steps[:-1]
        last_step_premises = last_step.split("->")[0].strip()
        last_step_formatted = f"{last_step_premises} -> hypothesis"
        prev_steps.append(last_step_formatted)
        full_proof = f"$proof$ = {'; '.join(prev_steps)};"
        return full_proof

    def preprocess_entailmentbank_example(self, example: Dict):
        self.context = self.process_identified_context(example['context'])
        self.hypothesis = example['hypothesis']

    def preprocess_ruletaker_example(self, example: Dict):
        self.context = self.process_identified_context(example['context'])
        self.hypothesis = example['hypothesis']

    def has_reached_hypothesis(self, threshold=0.8) -> bool:
        last_proof = self.get_last_proof()
        if last_proof is None:
            return False
        with torch.no_grad():
            inp = self.bleurt_tokenizer([self.hypothesis], [last_proof], return_tensors='pt')
            score = self.bleurt_model(**inp)[0].squeeze().item()
        # print("Bleurt score: ", score)

        return score > threshold

    def add_proof_step(self, proof: str, step: FrozenSet[str], score: float):
        next_int_id = self.num_inferred_nodes + 1
        self.intermediates[f'int{next_int_id}'] = proof
        self.context[f'int{next_int_id}'] = proof
        step_txt = " & ".join(step)
        self.proof_steps.append(f"{step_txt} -> int{next_int_id}: {proof}")

        # assuming the scores are log likelihoods
        self.score += score

    def get_selection_prompt(self) -> str:
        prefix = "select the best steps for induction in forward reasoning from the following premises:\n"
        premises = "\n".join([f"{k}: {v}" for k, v in self.context.items()])
        context = f"premises:\n{premises}\nhypothesis:\n{self.hypothesis}\nproof steps: "
        return f"{prefix}{context}"

    def get_deduction_prompt(self, next_step: FrozenSet[str], deductor_add_hyp=False) -> str:
        instruction = "given premises perform an induction step:\n"

        if deductor_add_hyp:
            prefix = f"{instruction}hypothesis: {self.hypothesis}\ninduction:\n"
        else:
            prefix = instruction

        step_texts = []

        for step in next_step:
            try:
                step_texts.append(self.context[step].strip())
            except KeyError as e:
                raise KeyError(f"Step {step} not found in context")
        if len(step_texts) == 0:
            raise ValueError("No steps were given to deduce")

        input_txt = prefix + " [AND] ".join(step_texts) + " [INFER]"
        return input_txt

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
        sents_d = OrderedDict()
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
                 top_k: int = 20) -> \
        Tuple[List[FrozenSet[str]], List[float]]:
    """
    Sample the next steps to take in the proof search. It will return a list which each element is a list of
    identifiers of the steps to take. Also returns a list of scores for each of the sampled steps.
    """
    input_txt = search_state.get_selection_prompt()

    print("selection prompt:")
    pprint(input_txt)

    inputs = selector_tokenizer(input_txt, return_tensors='pt')
    inputs['input_ids'] = inputs['input_ids'].to(selector.device)
    inputs['attention_mask'] = inputs['attention_mask'].to(selector.device)
    outputs = selector.generate(**inputs, max_length=100, num_return_sequences=top_k, num_beams=top_k,
                                do_sample=False, return_dict_in_generate=True, output_scores=True, )

    step_texts = selector_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    scores = outputs.sequences_scores
    samples = []
    sample_scores = []

    for step_text, score in zip(step_texts, scores):
        step_ids = process_generated_steps(step_text)
        if len(step_ids) == 0:
            raise ValueError("No steps were generated")
        if step_ids not in samples:
            samples.append(step_ids)
            sample_scores.append(score)
            print(f"sample: {step_text}, score: {score}")

    assert len(samples) > 0, "No valid steps were sampled"
    return samples, sample_scores


def batch_apply_deductor(deductor, deductor_tokenizer, search_states: List[SearchState],
                         next_steps: List[FrozenSet[str]], scores: List[float], num_beams=5,
                         deductor_add_hyp=False) -> List[SearchState]:
    assert len(search_states) == len(next_steps)

    valid_search_states = []
    valid_scores = []
    input_txts = []
    for s, next_step, score in zip(search_states, next_steps, scores):
        try:
            # avoid terminating search if there are bad steps given by the selector
            deduction_prompt = s.get_deduction_prompt(next_step, deductor_add_hyp=deductor_add_hyp)
        except Exception as e:
            print(e)
            continue
        input_txts.append(deduction_prompt)
        valid_search_states.append(s)
        valid_scores.append(score)
    if len(input_txts) == 0:
        print("No valid deduction prompts were generated")
        return search_states

    generated_texts, _ = batched_generate(input_txts, deductor, deductor_tokenizer, max_length=100,
                                          num_return_sequences=1,
                                          num_beams=num_beams)
    # inputs = deductor_tokenizer(input_txts, return_tensors='pt', padding='longest', truncation=True)
    # outputs = deductor.generate(**inputs, max_length=100, num_return_sequences=1, num_beams=num_beams)
    # generated_texts = deductor_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for i, (search_state, generated_text, score) in enumerate(zip(valid_search_states, generated_texts, valid_scores)):
        search_state.add_proof_step(generated_text, next_steps[i], score=score)

    return search_states


def apply_deductor(deductor, deductor_tokenizer, search_state: SearchState, next_step: FrozenSet[str], score: float) \
        -> SearchState:
    """
    Apply the deductor model to the current proof state and the next steps to take. It will return the updated search
    state with the newly created node.
    """

    input_txt = search_state.get_deduction_prompt(next_step)

    print(f"deductor input: {input_txt}")
    inputs = deductor_tokenizer(input_txt, return_tensors='pt')
    outputs = deductor.generate(**inputs, max_length=100, num_return_sequences=1, num_beams=5)
    generated_text = deductor_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    search_state.add_proof_step(generated_text, next_step, score)
    return search_state


def greedy_proof_search(example: Dict, deductor: PreTrainedModel, deductor_tokenizer: PreTrainedTokenizer,
                        selector: PreTrainedModel, selector_tokenizer: PreTrainedTokenizer,
                        bleurt_tokenizer: PreTrainedTokenizer,
                        bleurt_model: PreTrainedModel,
                        n_iters: int = 5) -> str:
    search_state = SearchState(example, bleurt_model, bleurt_tokenizer)
    iterations = 0
    print(search_state)
    try:
        while not search_state.has_reached_hypothesis() and iterations < n_iters:
            pprint(f"step {iterations + 1}")
            next_steps, scores = sample_steps(selector, selector_tokenizer, search_state)
            next_step = next_steps[0]
            score = scores[0]

            print("Next step: ", next_step)
            search_state = apply_deductor(deductor, deductor_tokenizer, search_state, next_step, score)
            pprint("Current proof step: " + search_state.proof_steps[-1])
            iterations += 1

        for proof_step in search_state.proof_steps:
            pprint(proof_step)
    except Exception as e:
        # print("Failed to generate proof for this example!")
        print(e)
        raise
    return search_state.get_formatted_full_proof()


def batched_generate(inputs, model, tokenizer, max_length=100, num_return_sequences=1, num_beams=5, batch_size=1):
    outputs = []
    scores = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        batch_inputs = tokenizer(batch, return_tensors='pt', padding='longest', truncation=True)
        batch_inputs['input_ids'] = batch_inputs['input_ids'].to(model.device)
        batch_inputs['attention_mask'] = batch_inputs['attention_mask'].to(model.device)
        batch_outputs = model.generate(**batch_inputs, max_length=max_length, num_return_sequences=num_return_sequences,
                                       num_beams=num_beams, do_sample=False, return_dict_in_generate=True,
                                       output_scores=True)
        output_texts = tokenizer.batch_decode(batch_outputs.sequences, skip_special_tokens=True)
        if num_beams > 1:
            scores.extend(batch_outputs.sequences_scores)
        else:
            # todo: quick fix to the problem of none sequence scores when num_beams=1
            scores.extend([0.0] * len(output_texts))
        outputs.extend(output_texts)

    if isinstance(scores, list):
        scores = torch.tensor(scores)
    return outputs, scores


def batch_sample_steps(selector: PreTrainedModel, selector_tokenizer: PreTrainedTokenizer,
                       search_states: List[SearchState], top_k: int = 20) -> List[List[Tuple[FrozenSet, float]]]:
    """
    takes a list of search_states and for each of them calculates topk next steps. After post-processing each next step
    and removing redundant ones, returns a list with the size of the number of input search states in which each element
    is a list of top outputs along with their scores
    """
    step_texts, scores = batched_generate([s.get_selection_prompt() for s in search_states], selector,
                                          selector_tokenizer,
                                          max_length=100, num_return_sequences=top_k, num_beams=top_k)
    output_idx = 0
    outputs_splited = []

    while output_idx < len(search_states):
        idx = output_idx * top_k
        samples = []
        sample_scores = []
        for i in range(idx, idx + top_k):
            step_text, score = step_texts[i], scores[i]
            step_ids = process_generated_steps(step_text)

            if step_ids not in samples:
                samples.append(step_ids)
                sample_scores.append(score.item())

        outputs_splited.append(list(zip(samples, sample_scores)))
        output_idx += 1

    return outputs_splited


def batch_test_if_hypothesis_reached(search_states: List[SearchState], bleurt_tokenizer: PreTrainedTokenizer,
                                     bleurt_model: PreTrainedModel, threshold: float = 0.6) -> Tuple[
    List[bool], List[float]]:
    last_proofs = [s.get_last_proof() for s in search_states]
    valid_proof_indices = [i for i, proof in enumerate(last_proofs) if proof is not None]
    valid_last_proofs = [last_proofs[i] for i in valid_proof_indices]
    hypotheses = [s.hypothesis for s in search_states]
    hypotheses = [hypotheses[i] for i in valid_proof_indices]

    result = [False] * len(search_states)
    if len(valid_last_proofs) == 0:
        return result, [0.0] * len(search_states)

    with torch.no_grad():
        inp = bleurt_tokenizer(hypotheses, valid_last_proofs, return_tensors='pt', padding='longest', truncation=True)
        inp['input_ids'] = inp['input_ids'].to(bleurt_model.device)
        inp['attention_mask'] = inp['attention_mask'].to(bleurt_model.device)
        inp['token_type_ids'] = inp['token_type_ids'].to(bleurt_model.device)
        scores = bleurt_model(**inp)[0].squeeze().detach().tolist()
        if isinstance(scores, float):
            scores = [scores]
    # print("BLEURT scores: ", scores)
    reached = [score > threshold for score in scores]
    for i, r in zip(valid_proof_indices, reached):
        result[i] = r
    return result, scores


def proof_beam_search(example: Dict, deductor: PreTrainedModel, deductor_tokenizer: PreTrainedTokenizer,
                      selector: PreTrainedModel, selector_tokenizer: PreTrainedTokenizer,
                      bleurt_tokenizer: PreTrainedTokenizer,
                      bleurt_model: PreTrainedModel,
                      n_iters: int = 5, n_search_beams: int = 2,
                      deductor_add_hyp: bool = False, verbose=False,
                      hypothesis_acceptance_threshold: float = 0.6) -> str:
    initial_search_state = SearchState(example, bluert_model=bleurt_model, bluert_tokenizer=bleurt_tokenizer)
    active_beams = [initial_search_state]
    reached_hypothesis = []
    print(initial_search_state)
    try:
        for i in range(n_iters):
            reached, hyp_scores = batch_test_if_hypothesis_reached(active_beams, bleurt_tokenizer, bleurt_model,
                                                                   threshold=hypothesis_acceptance_threshold)
            to_terminate = []
            for r, hscore, beam in zip(reached, hyp_scores, active_beams):
                if r:
                    reached_hypothesis.append((beam, hscore))
                    to_terminate.append(beam)
            for beam in to_terminate:
                active_beams.remove(beam)

            if len(active_beams) == 0:
                break

            outputs = batch_sample_steps(selector, selector_tokenizer, active_beams, top_k=n_search_beams)
            flattened_outputs = []

            cur_beam_scores = [b.score for b in active_beams]

            for idx, (out, cur_score) in enumerate(zip(outputs, cur_beam_scores)):
                for step, score in out:
                    # idx, step, cum_score, step_score
                    flattened_outputs.append((idx, step, cur_score + score, score))

            flattened_outputs = sorted(flattened_outputs, key=lambda x: -x[2])[:n_search_beams]
            active_beams = [duplicate_search_state(active_beams[beam_id]) for beam_id, _, _, _ in flattened_outputs]
            next_steps = [o[1] for o in flattened_outputs]
            scores = [o[3] for o in flattened_outputs]
            print("new scores: ", scores)

            active_beams = batch_apply_deductor(deductor, deductor_tokenizer, search_states=active_beams,
                                                next_steps=next_steps, scores=scores, deductor_add_hyp=deductor_add_hyp)

            if verbose:
                print(f"currently {len(reached_hypothesis)} beams reached the hypothesis")
                print(f"iteration {i} top {n_search_beams} beams:")
                for (_, step, cum_score, _), beam in zip(flattened_outputs, active_beams):
                    print(f"step: {step} cum_score: {cum_score} proof step: {beam.proof_steps[-1]}")


    except Exception as e:
        print("Failed to generate proof for this example!")
        print(e)
        raise

    print("top k proofs:")

    if len(reached_hypothesis) == 0:
        print("none of the beams reached the hypothesis")
        for beam in active_beams:
            pprint(beam.get_formatted_full_proof())
        return active_beams[0].get_formatted_full_proof()
    else:
        # sort results by score:
        # reached_hypothesis = sorted(reached_hypothesis, key=lambda x: -x.score)[:n_search_beams]
        # sort results by how close they are to the hypothesis
        reached_hypothesis = sorted(reached_hypothesis, key=lambda x: -x[0].score)[:n_search_beams]

        for beam, hscore in reached_hypothesis:
            pprint(f"proof score: {beam.score} hypothesis match score: {hscore}\n{beam.get_formatted_full_proof()}")

        reached_hypothesis = [x[0] for x in reached_hypothesis]
        return reached_hypothesis[0].get_formatted_full_proof()


def early_selection_weighted_search(example: Dict, deductor: PreTrainedModel, deductor_tokenizer: PreTrainedTokenizer,
                                    selector: PreTrainedModel, selector_tokenizer: PreTrainedTokenizer,
                                    bleurt_tokenizer: PreTrainedTokenizer,
                                    bleurt_model: PreTrainedModel,
                                    n_iters: int = 6, n_search_beams: int = 2,
                                    hypothesis_acceptance_threshold: float = 0.6,
                                    deductor_add_hyp: bool = False, verbose=False) -> str:
    initial_search_state = SearchState(example, bluert_model=bleurt_model, bluert_tokenizer=bleurt_tokenizer)

    active_beams = [initial_search_state]
    reached_hypothesis = []

    if verbose:
        print(initial_search_state)

    try:
        for i in range(n_iters):
            reached, hyp_scores = batch_test_if_hypothesis_reached(active_beams, bleurt_tokenizer, bleurt_model,
                                                                   threshold=hypothesis_acceptance_threshold)
            to_terminate = []
            for r, hscore, beam in zip(reached, hyp_scores, active_beams):
                if r:
                    reached_hypothesis.append((beam, hscore))
                    to_terminate.append(beam)
            for beam in to_terminate:
                active_beams.remove(beam)

            if len(active_beams) == 0:
                break

            if i > 0:
                outputs = batch_sample_steps(selector, selector_tokenizer, active_beams, top_k=1)
            else:
                outputs = batch_sample_steps(selector, selector_tokenizer, active_beams, top_k=n_search_beams)

            flattened_outputs = []
            cur_beam_scores = [b.score for b in active_beams]

            for idx, (out, cur_score) in enumerate(zip(outputs, cur_beam_scores)):
                for step, score in out:
                    flattened_outputs.append((idx, step, cur_score + score, score))

            flattened_outputs = sorted(flattened_outputs, key=lambda x: -x[2])[:n_search_beams]

            active_beams = [duplicate_search_state(active_beams[beam_id]) for beam_id, _, _, _ in flattened_outputs]
            next_steps = [o[1] for o in flattened_outputs]
            scores = [o[3] for o in flattened_outputs]
            # print(list(zip(next_steps, scores)))
            active_beams = batch_apply_deductor(deductor, deductor_tokenizer, search_states=active_beams,
                                                next_steps=next_steps, scores=scores, deductor_add_hyp=deductor_add_hyp)

            if verbose:
                print(f"iteration {i} top {n_search_beams} beams:")
                for (_, step, cum_score, _), beam in zip(flattened_outputs, active_beams):
                    print(f"step: {step} score: {cum_score} proof step: {beam.proof_steps[-1]}")

    except Exception as e:
        print("Failed to generate proof for this example!")
        print(e)
        raise

    print("top k proofs:")

    if len(reached_hypothesis) == 0:
        print("none of the beams reached the hypothesis")
        for beam in active_beams:
            pprint(beam.get_formatted_full_proof())
        return active_beams[0].get_formatted_full_proof()
    else:
        # sort results by score:
        reached_hypothesis = sorted(reached_hypothesis, key=lambda x: -x[0].score)[:n_search_beams]

        for beam, hscore in reached_hypothesis:
            pprint(f"proof score: {beam.score} hypothesis match score: {hscore}\n{beam.get_formatted_full_proof()}")

        reached_hypothesis = [x[0] for x in reached_hypothesis]
        return reached_hypothesis[0].get_formatted_full_proof()


def duplicate_search_state(search_state: SearchState) -> SearchState:
    new_search_state = SearchState(None, search_state.bleurt_model, search_state.bleurt_tokenizer)
    new_search_state.context = deepcopy(search_state.context)
    new_search_state.intermediates = deepcopy(search_state.intermediates)
    new_search_state.proof_steps = deepcopy(search_state.proof_steps)
    new_search_state.hypothesis = search_state.hypothesis
    new_search_state.score = search_state.score
    return new_search_state


def pprint(txt: str):
    print("====================================")
    print(txt)
    print("====================================")


def load_examples(test_data_path: str):
    with open(test_data_path, 'r') as f:
        examples = f.readlines()
    return [json.loads(example) for example in examples]


def load_model_and_tokenizer(model_name: str, cache_dir=None, load_in_8bit=False):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, load_in_8bit=load_in_8bit)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer


def run(deductor_path: str, selector_path: str, test_data_path: str, output_dir: str, cache_dir=None,
        n_search_beams=1, hypothesis_acceptance_threshold=0.6, deductor_add_hyp=False, verbose=False,
        search_algorithm='early_selection'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deductor, deductor_tokenizer = load_model_and_tokenizer(deductor_path, cache_dir=cache_dir)
    deductor = deductor.to(device)
    deductor.eval()
    selector, selector_tokenizer = load_model_and_tokenizer(selector_path, cache_dir=cache_dir)
    selector.eval()
    selector = selector.to(device)

    bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512", cache_dir=cache_dir)
    bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512",
                                                                      cache_dir=cache_dir)
    bleurt_model.eval()
    bleurt_model = bleurt_model.to(device)

    examples = load_examples(test_data_path)
    # np.random.shuffle(examples)

    results = []
    for example in tqdm(examples):

        # todo: make sure false examples are not in dataset
        if 'depth' in example and (example['depth'] is None or example['depth'] < 1 or example['answer'] is False):
            continue

        print(f"using search algorithm: {search_algorithm}")
        if search_algorithm == 'greedy':
            proof = greedy_proof_search(example, deductor, deductor_tokenizer, selector, selector_tokenizer,
                                        bleurt_tokenizer, bleurt_model, cache_dir)
        elif search_algorithm == 'beam_search':
            proof = proof_beam_search(example, deductor, deductor_tokenizer, selector, selector_tokenizer,
                                      bleurt_tokenizer, bleurt_model, n_search_beams=n_search_beams,
                                      hypothesis_acceptance_threshold=hypothesis_acceptance_threshold,
                                      verbose=verbose, deductor_add_hyp=deductor_add_hyp)
        elif search_algorithm == 'early_selection':
            proof = early_selection_weighted_search(example, deductor, deductor_tokenizer, selector, selector_tokenizer,
                                                    bleurt_tokenizer, bleurt_model, n_search_beams=n_search_beams,
                                                    hypothesis_acceptance_threshold=hypothesis_acceptance_threshold,
                                                    deductor_add_hyp=deductor_add_hyp, verbose=verbose)
        else:
            raise ValueError(f"Unknown search algorithm: {search_algorithm}")

        results.append(proof)

        try:
            print(visualize_tree(proof, example))
        except Exception as e:
            print(f"Failed to visualize tree: {e}")

        if verbose:
            print("*" * 50)
            print("Ground truth proof:")
            print(example['proof'] if 'proof' in example else example['proofs'])
            print("*" * 50)
            # waiting on user for debug purposes
            input()

    os.makedirs(output_dir, exist_ok=True)
    output_file_name = test_data_path.split('/')[-1]
    with open(os.path.join(output_dir, output_file_name.replace('.json', '.tsv')), 'w') as f:
        f.write("\n".join(results))


if __name__ == '__main__':
    Fire(run)

# to run locally
# run(
#     deductor_path='models/deductor-t5-large',
#     selector_path='models/selector-flant5-large',
#     test_data_path='data/test/proof-d3.jsonl',
#     output_dir='results',
#     n_search_beams=2
# )
