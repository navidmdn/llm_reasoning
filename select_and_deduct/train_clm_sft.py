import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, LlamaForCausalLM, AutoConfig, EarlyStoppingCallback
import numpy as np
from trl.commands.cli_utils import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
import nltk
from select_and_deduct.prepare_clm_dataset import RESPONSE_BEGIN_TOKENS, INSTRUCTION_BEGIN_TOKENS
from trl import setup_chat_format, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from evaluate_selector import permutation_invariant_metrics

from trl import (
    SFTConfig,
    SFTTrainer)

import evaluate

import codecs


# Comment in if you want to use the Llama 3 instruct template but make sure to add modules_to_save
# LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

#todo: test fsdp:
# ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./scripts/run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml

@dataclass
class ScriptArguments:
    train_file: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    dev_file: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    model_id: str = field(
        default="stas/tiny-random-llama-2", metadata={"help": "Model ID to use for SFT training"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Whether to load the model in 4bit"}
    )
    use_peft: bool = field(
        default=False, metadata={"help": "Whether to use PEFT"}
    )
    flash_attention: bool = field(
        default=False, metadata={"help": "Whether to use fast attention"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Path to the cache directory"}
    )
    max_eval_samples: int = field(
        default=None, metadata={"help": "Max number of samples to evaluate"}
    )
    num_return_sequences: int = field(
        default=1, metadata={"help": "Number of sequences to generate"}
    )
    local_test: bool = field(
        default=False, metadata={"help": "Whether to run a local test"}
    )
    max_new_tokens: int = field(
        default=100, metadata={"help": "Max number of tokens to generate"}
    )


def load_test_llama3_tokenizer_and_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    config.hidden_size = 256
    config.intermediate_size = 16
    config.num_hidden_layers = 2
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    model = LlamaForCausalLM(config=config)
    return tokenizer, model



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def training_function(script_args, training_args):
    ################
    # Dataset
    ################

    train_dataset = load_dataset(
        "json",
        data_files=script_args.train_file,
        split="train",
        cache_dir=script_args.cache_dir,

    )
    dev_dataset = load_dataset(
        "json",
        data_files=script_args.dev_file,
        split="train",
        cache_dir=script_args.cache_dir,
    )

    if script_args.max_eval_samples is not None:
        n_samples = min(script_args.max_eval_samples, len(dev_dataset))
        print(f"Limiting evaluation samples to {n_samples}")
        dev_dataset = dev_dataset.shuffle().select(range(n_samples))

    ################
    # Model & Tokenizer
    ################

    # Tokenizer
    if script_args.local_test:
        tokenizer, model = load_test_llama3_tokenizer_and_model(script_args.model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True, cache_dir=script_args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    # padding side should be left for causal language modeling
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    instruction_template = tokenizer.encode(INSTRUCTION_BEGIN_TOKENS, add_special_tokens=False)[1:]
    response_template = tokenizer.encode(RESPONSE_BEGIN_TOKENS, add_special_tokens=False)[1:]

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        response_template=response_template,
    )

    # print random sample
    with training_args.main_process_first(
            desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    # Model
    if 'llama' in script_args.model_id.lower():
        #todo: checkout official example code for correct dtype
        print("Using llama 3 recommended data type")
        torch_dtype = torch.bfloat16
        quant_storage_dtype = torch.bfloat16
    else:
        torch_dtype = None
        quant_storage_dtype = None

    quantization_config = None
    if script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    if not script_args.local_test:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2" if script_args.flash_attention else None, #todo: make it configurable; use sdpa, alternatively use "flash_attention_2"
            torch_dtype=quant_storage_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            cache_dir=script_args.cache_dir,
        )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    peft_config = None
    if script_args.use_peft:
        # LoRA config based on QLoRA paper & Sebastian Raschka experiment
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=64,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
        )

    metric = evaluate.load("rouge", cache_dir=script_args.cache_dir)
    def compute_metrics(eval_predictions):
        inputs = eval_predictions.inputs
        labels = eval_predictions.label_ids
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        inputs = np.where(inputs == -100, tokenizer.pad_token_id, inputs)

        # separate the instruction and response
        input_txts = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        instructions = [txt.split(RESPONSE_BEGIN_TOKENS)[0].strip() for txt in input_txts]

        inputs = tokenizer(instructions, return_tensors="pt", padding='longest')
        inputs['input_ids'] = inputs['input_ids'].to(model.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(model.device)

        input_lens = inputs["input_ids"].shape[1]
        #todo: configure generation configs specifically for selection task
        outputs = model.generate(
            **inputs,
            max_new_tokens=script_args.max_new_tokens,
            num_return_sequences=script_args.num_return_sequences,
            num_beams=script_args.num_return_sequences,
            do_sample=False,
        )
        outputs = outputs[:, input_lens:]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_preds = [pred.split('\n')[0] for pred in decoded_preds]
        for pred, label in zip(decoded_preds, decoded_labels):
            print(f"Pred: {pred}")
            print(f"Label: {label}")
            print("------")

        perm_inv_metrics = {}
        if script_args.num_return_sequences > 1:
            perm_inv_metrics = permutation_invariant_metrics(decoded_preds, decoded_labels,
                                                             script_args.num_return_sequences)
            decoded_preds = decoded_preds[0::script_args.num_return_sequences]

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [len(pred) for pred in decoded_preds]
        result["gen_len"] = np.mean(prediction_lens)
        result.update(**perm_inv_metrics)
        return result


    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        max_seq_length=2048,
        dataset_text_field="text",
        eval_dataset=dev_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )
    if trainer.accelerator.is_main_process and script_args.use_peft:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()

    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)