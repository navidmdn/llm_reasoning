#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from random import random

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import numpy as np
import transformers
from transformers import Trainer
from dataset_handler import DataArguments, make_supervised_data_module, smart_tokenizer_and_embedding_resize, \
    DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
import evaluate

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from typing import List
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
from transformers.trainer_pt_utils import IterableDatasetShard, nested_concat
from transformers.trainer import has_length, find_batch_size, denumpify_detensorize
from tqdm import tqdm
from transformers import StoppingCriteria

rouge_metric = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


class CustomTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

    @staticmethod
    def get_average_metrics(metrics_list: List[Dict]):
        metrics = {}
        for key in metrics_list[0].keys():
            metrics[key] = np.mean([x[key] for x in metrics_list])
        return metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        batch_size = self.args.eval_batch_size
        eval_dataset = getattr(dataloader, "dataset", None)

        preds_all = []
        labels_all = []
        losses_all = []
        all_metrics = []
        num_samples = 0

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)

        observed_num_examples = 0

        num_batches = num_samples//batch_size

        for step, inputs in tqdm(enumerate(dataloader), total=num_batches):

            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            with torch.no_grad():
                cur_preds = (torch.argmax(logits, dim=-1).detach().cpu().numpy())
                preds_all.append(cur_preds)
                cur_labels = labels.detach().cpu().numpy()
                labels_all.append(cur_labels)
                losses_all.append(loss.detach().cpu().numpy())

                cur_metrics = self.compute_metrics(EvalPrediction(predictions=cur_preds, label_ids=cur_labels))
                all_metrics.append(cur_metrics)

        metrics = self.get_average_metrics(all_metrics)
        metrics[f"{metric_key_prefix}_loss"] = np.mean(losses_all)
        metrics = denumpify_detensorize(metrics)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        return EvalLoopOutput(
            predictions=nested_concat([], preds_all, padding_index=-100),
            label_ids=nested_concat([], labels_all, padding_index=-100),
            metrics=metrics,
            num_samples=num_samples)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    def compute_metrics(eval_preds):
        result = {}
        preds, labels = eval_preds

        # removed the token distribution in custom eval loop so don't need the following line
        # preds = np.argmax(preds, axis=-1)

        # todo: stop after eos token: isn't there a better way to do it?
        preds_list = []
        for p, l in zip(preds, labels):
            # last ignore token is the first prediction
            label_start = max(0, np.where(l != -100)[0][0]-1)
            p = p[label_start:]
            if tokenizer.eos_token_id in p:
                first_pad_idx = np.where(p == tokenizer.eos_token_id)
                if len(first_pad_idx) > 0 and first_pad_idx[0][0] < len(p)-1:
                    p[first_pad_idx[0][0]+1:] = tokenizer.pad_token_id
            preds_list.append(p)

        decoded_preds = tokenizer.batch_decode(preds_list, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        exact_match_result = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)
        if random() < 0.01:
            print(list(zip(decoded_labels, decoded_preds))[:5])
        generated_pred = np.where(labels == tokenizer.pad_token_id, 0, preds)
        prediction_lens = [np.count_nonzero(pred) for pred in generated_pred]
        rouge_result["gen_len"] = np.mean(prediction_lens)

        result.update(rouge_result)
        result.update(exact_match_result)

        return result

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    #todo: custom evaluation loop required to speed up evaluation
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics=compute_metrics, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
