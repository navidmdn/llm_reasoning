import fire
from utils.hf_inference import load_model_and_tokenizer, generate_resp
from utils.misc import load_prompt
import torch

sys1_gen_prompt = """
Here is my thought process:

{thought_process}

Here is my detailed response:
"""


def run_two_system(s1_path: str, s2_path: str, cache_dir: str = None, qlora=True,
                   s1_sys_prompt_path: str = 'prompts/system1_sys_prompt.txt',
                   s2_sys_prompt_path: str = 'prompts/system2_sys_prompt.txt') -> None:
    s1_model, s1_tokenizer = load_model_and_tokenizer(s1_path, cache_dir=cache_dir, qlora=qlora)
    s2_model, s2_tokenizer = load_model_and_tokenizer(s2_path, cache_dir=cache_dir, qlora=qlora)

    s1_system_msg = load_prompt(s1_sys_prompt_path)
    s2_system_msg = load_prompt(s2_sys_prompt_path)
    print("loaded models and tokenizers")

    instruction = input("Enter the instruction: ")
    # instruction = "write a python code to print fibonacci series"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    thought_messages = []
    messages = []
    while len(instruction) != 0:

        thought_messages.append({'role': 'user', 'content': instruction})
        thought = generate_resp(s2_model, s2_tokenizer, s2_system_msg,
                                [{'messages': thought_messages}],
                                device=device, temperature=0.8, top_p=0.9, max_new_tokens=256, do_sample=True)[0]
        thought_messages.append({'role': 'assistant', 'content': thought})

        print("thought process: ", thought)
        prompt = sys1_gen_prompt.format(thought_process=thought)
        messages.append({'role': 'user', 'content': instruction})
        messages.append({'role': 'assistant', 'content': prompt})
        response = generate_resp(s1_model, s1_tokenizer, s1_system_msg, [{'messages': messages}],
                                 device=device, temperature=0.8, top_p=0.9, max_new_tokens=256, do_sample=True,
                                 add_generation_prompt=False)[0]
        messages[-1] = {'role': 'assistant', 'content': messages[-1]['content'] + response}
        print("response: ", response)

        instruction = input("Enter the instruction: ")

        if instruction == 'r':
            instruction = input("Enter the instruction: ")
            thought_messages = []
            messages = []
            continue


if __name__ == '__main__':
    fire.Fire(run_two_system)