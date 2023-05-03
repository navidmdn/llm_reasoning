import json
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from datasets import Dataset, load_dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# from datasets import disable_caching
# disable_caching()

@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    ruletaker_dev_path: str = field(default=None, metadata={"help": "Path to the ruletaker dev dataset."})
    entailmenttree_dev_path: str = field(default=None, metadata={"help": "Path to the entailmenttree dev datasets."})


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def load_supervised_dataset(tokenizer, data_files) -> Dataset:
    dataset = load_dataset("json", data_files=data_files)

    def change_premise(example):
        example['premises'] = [e + ' [THEREFORE], ' for e in example['premises']]
        example['conclusion'] = [e + tokenizer.eos_token for e in example['conclusion']]
        return example

    dataset = dataset.map(change_premise, batched=True, batch_size=256, num_proc=4)

    def preprocess(example):
        """Preprocess the data by tokenizing."""
        full_txt = [s + t for s, t in zip(example['premises'], example['conclusion'])]
        examples_tokenized = _tokenize_fn(full_txt, tokenizer)
        sources_tokenized = _tokenize_fn(example['premises'], tokenizer)
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        #     label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    dataset = dataset.map(preprocess, batched=True, num_proc=8, batch_size=256,
                          remove_columns=['premises', 'conclusion'])

    return dataset


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_collator = transformers.DataCollatorForTokenClassification(tokenizer=tokenizer,
                        padding="longest", max_length=tokenizer.model_max_length, label_pad_token_id=IGNORE_INDEX)

    data_files = {
        'train': data_args.train_data_path,
    }

    if data_args.ruletaker_dev_path is not None:
        data_files.update({'ruletaker_dev': data_args.ruletaker_dev_path})
    if data_args.entailmenttree_dev_path is not None:
        data_files.update({'entailmenttree_dev': data_args.entailmenttree_dev_path})

    dataset = load_supervised_dataset(tokenizer=tokenizer, data_files=data_files)

    processed_ds = dict(
        train_dataset=dataset['train'],
        eval_dataset={k: dataset[k] for k in dataset if 'dev' in k},
        data_collator=data_collator)

    return processed_ds



