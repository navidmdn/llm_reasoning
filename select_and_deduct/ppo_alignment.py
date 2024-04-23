# imports
import torch
from transformers import AutoTokenizer
from transformers import top_k_top_p_filtering

from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
import json
import numpy as np
from typing import List
from select_and_deduct.evaluate_selector import _permutation_invariant_match
from torch import nn
import torch.nn.functional as F
from fire import Fire
from tqdm import tqdm

def get_reward(preds: List[str], targets: List[str]) -> List[torch.FloatTensor]:
    rewards = []
    for pred, target in zip(preds, targets):
        print(f"pred: {pred}, target: {target}")
        match = _permutation_invariant_match(pred, target)
        rewards.append(torch.FloatTensor([float(match)]))
    return rewards

def respond_to_batch(
    model: nn.Module, queries: List[torch.LongTensor], targets: List[torch.LongTensor] = None, txt_len: int = 20,
        top_k: int = 0, top_p: float = 1.0
) -> torch.LongTensor:
    """Sample text from language model."""
    input_ids = queries
    decoder_input_ids = torch.LongTensor([[0] for _ in range(input_ids.shape[0])]).to(input_ids.device)

    for _i in range(txt_len):
        # Get Logits
        #todo: do i need to pass decoder input ids?
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=-1)
    return decoder_input_ids[:, -txt_len:]


def run(model_name: str, train_path: str, batch_size: int = 2, n_iters: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name)
    model = model.to(device)
    model_ref = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # initialize trainer
    ppo_config = PPOConfig(batch_size=batch_size, mini_batch_size=batch_size)

    with open(train_path, 'r') as f:
        examples = f.readlines()
        examples = [json.loads(example) for example in examples]

    # get a batch of examples
    reward_window = []

    for i in tqdm(range(n_iters)):
        batch = np.random.choice(examples, batch_size)
        prefix = 'select the best steps for induction in forward reasoning from the following premises:\n'
        inputs = [f"{prefix}{example['premises']}" for example in batch]
        targets = [example['steps'] for example in batch]

        query_tensor = tokenizer(inputs, return_tensors="pt", padding='longest', truncation=True)['input_ids']
        target_ids = tokenizer(targets, return_tensors="pt", padding='longest', truncation=True)['input_ids']
        query_tensor = query_tensor.to(device)
        target_ids = target_ids.to(device)

        # get model response
        response_tensor = respond_to_batch(model, query_tensor, target_ids)
        response_texts = tokenizer.batch_decode(response_tensor, skip_special_tokens=False)
        # todo: stop at eos for each example

        rewards = get_reward(response_texts, targets)

        reward_window.extend([r.item() for r in rewards])

        # create a ppo trainer
        ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

        # train model for one step with ppo
        train_stats = ppo_trainer.step(list(query_tensor), list(response_tensor), rewards)

        if i % 10 == 0:
            print(f"iter {i}: avg reward {np.mean(reward_window)}")
            reward_window = []


if __name__ == '__main__':
    # run('t5-small', '../data/selector_train_merged.json')

    Fire(run)

