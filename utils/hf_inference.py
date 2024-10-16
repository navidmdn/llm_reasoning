from fire import Fire
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch


def load_model_and_tokenizer(model_name_or_path, cache_dir, qlora=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if qlora:
        torch_dtype = torch.bfloat16
        quant_storage_dtype = torch.bfloat16
        bnb_4bit_use_double_quant = True
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=quant_storage_dtype,
            cache_dir=cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    return model, tokenizer

def pprint(prefix, txt):
    print(f"===================={prefix}===================")
    print(txt)
    print("========================================")


def generate_resp(model, tokenizer, system_msg,
                  examples, device, temperature=0.1, top_p=0.9,
                  max_new_tokens=1024, do_sample=True, add_generation_prompt=True):

    prompts = [[
        {"role": "system", "content": system_msg},
        *ex['messages'],
    ] for ex in examples]

    continue_final_message = not add_generation_prompt
    #todo: continue_final_message didnt work as it also created a <eot> token at the end
    input_chat_prompts = [tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tokenize=False,
    ) for prompt in prompts]

    # pprint("input_chat_prompt", input_chat_prompts[0])

    inputs = tokenizer(input_chat_prompts, return_tensors='pt', padding=True, truncation=False)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    input_ids = inputs['input_ids'].to(device)
    attention_masks = inputs['attention_mask'].to(device)

    #todo: manually removing the eot in case of continue_final_message
    if continue_final_message:
        assert len(input_ids) == 1
        if input_ids[0][-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>"):
            input_ids = input_ids[:, :-1]
            attention_masks = attention_masks[:, :-1]

    output = model.generate(
        input_ids,
        attention_mask=attention_masks,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    resps = tokenizer.batch_decode(output[:, input_ids.shape[1]:].cpu().numpy(), skip_special_tokens=True)
    return resps


