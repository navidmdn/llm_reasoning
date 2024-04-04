import os
from operator import itemgetter

# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
#
# with open('./secret', 'r') as f:
#     secret = f.read().strip()
#
# os.environ['LANGCHAIN_API_KEY'] = secret


import numpy as np
from llm_agents.utils import load_ollama_autoregressive_model, load_llamacpp_autoregressive_model, load_hf_auto_regressive_model
from llm_agents.data_handler import load_dataset, get_processed_entailmenet_dataset, get_processed_proofwriter_dataset, get_processed_entailmenet_dataset_textual
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotPromptTemplate
from fire import Fire
from langchain.prompts.example_selector import NGramOverlapExampleSelector, SemanticSimilarityExampleSelector


self_evaluation_examples = [
    {
        "deduction_step": "[BECAUSE] sent1: when water freezes , that water expands [AND] sent2: if there is a crack in a rock, then water can get into the crack [INFER] int1: if water freezes in the crack of the rock, then water will expand in the crack",
        "corrected_deduction_step": "int1: if water freezes in the crack of the rock, then water will expand in the crack",
    },
    {
        "deduction_step": "[BECAUSE] sent1: a river is a kind of a moving body of water [AND] sent2: most canyons are formed by flowing rivers through erosion over long periods of time [INFER] int1: most canyons are formed over long periods of time",
        "corrected_deduction_step": "int1: most canyons are formed by moving body of water through erosion over long periods of time",
    },
    {
        "deduction_step": "[BECAUSE] sent1: eyes are usually part of an animal for seeing [AND] sent5: an eagle is a kind of animal [INFER] int1: eagles see",
        "corrected_deduction_step": "int1: eyes are part of an eagle for seeing",
    },
    {
        "deduction_step": "[BECAUSE] int1: photosynthesis converts sunlight into sugar [AND] sent4: sunlight is a kind of solar energy [INFER] int2: solar energy is like sugar",
        "corrected_deduction_step": "int2: photosynthesis converts solar energy into sugar",
    }
]

def replace_conclusion(reasoning_step, corrected_conclusion):
    assert "[INFER]" in reasoning_step, "The reasoning step does not have an inference step"
    prems = reasoning_step.split("[INFER]")[0].strip()
    return f"{prems} [INFER] {corrected_conclusion}"


def run(llm_name='mistral:instruct', ollama=True, dataset='entailment_trees', cache_dir=None):

    if ollama:
        print("using ollama model")
        llm = load_ollama_autoregressive_model(model_name=llm_name)
    else:
        print("using huggingface model")
        llm = load_hf_auto_regressive_model(model_name=llm_name, load_in_4bit=True,
                                            cache_dir=cache_dir, max_len=100)

    if dataset == 'proof-writer':
        train_examples = load_dataset(f'./data/{dataset}/meta-train.jsonl')
        valid_examples = load_dataset(f'./data/{dataset}/meta-dev.jsonl')

        train_examples, valid_examples = get_processed_proofwriter_dataset(train_examples, valid_examples)

    elif dataset == 'entailment_trees':
        train_examples = load_dataset(f'./data/{dataset}/train-task1.jsonl')
        valid_examples = load_dataset(f'./data/{dataset}/dev-task1.jsonl')

        train_examples, valid_examples = get_processed_entailmenet_dataset_textual(train_examples, valid_examples)
    else:
        raise ValueError(f"Dataset {dataset} not found")

    example_prompt = PromptTemplate(
        input_variables=["context", "hypothesis", "proof"],
        template="premises:\n{context}\nhypothesis:\n{hypothesis}\nproof:\n{proof}\n"
    )

    eval_example_prompt = PromptTemplate(
        input_variables=["deduction_step", "corrected_deduction_step"],
        template="deduction_step:\n{deduction_step}\ncorrected_deduction_step:\n{corrected_deduction_step}\n"
    )

    for example in valid_examples:
        # selected_examples = example_selector.select_examples(example)
        # print(f"Examples most similar to the input {example} are: {selected_examples}")
        top_k = 5
        np.random.shuffle(train_examples)
        example_selector = NGramOverlapExampleSelector(
            examples=train_examples[:top_k],
            example_prompt=example_prompt,
        )

        prover_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="""You are a reasoning and logical prover. Your task is to generate a natural language proof to reach\
 a hypothesis. Here are a few examples and the format you need to follow to generate the proof. Write your proof in steps\
 and try to reach the hypothesis by building up step by step""",
            suffix="premises:\n{context}\nhypothesis:\n{hypothesis}\nproof:\n{proof}""",
            input_variables=["context", "hypothesis", "proof"]
        )

        self_evaluation_prompt = FewShotPromptTemplate(
            examples=self_evaluation_examples,
            example_prompt=eval_example_prompt,
            prefix="""Please do a precise logical deduction and check if the [INFER] sentence is correct. If the conclusion is correct
 write the sentence as it is, else write the correct sentence. examples:""",
            suffix="deduction_step:\n{deduction_step}\ncorrected_deduction_step:\n",
            input_variables=["deduction_step"],
        )

        def step_parser(output):
            first_step = output.split("[BECAUSE]")[1]
            return f"[BECAUSE] {first_step.strip()}"

        reasoning_step = ""
        gen_chain = prover_prompt | llm | step_parser | StrOutputParser()
        revision_chain = self_evaluation_prompt | llm | StrOutputParser()
        cur_ex = example.copy()
        cur_ex['proof'] = ""
        print(prover_prompt.format(**cur_ex))

        while 'hypothesis' not in reasoning_step:
            # todo: keep state of intermediate steps don't allow proofs with intermediate steps which are not proven yet!

            reasoning_step = gen_chain.invoke(cur_ex)
            print("\n==================Current Generated Proof==================")
            print(reasoning_step)
            print("================================================")

            revised_conclusion = revision_chain.invoke({'deduction_step': reasoning_step})
            revised_conclusion = revised_conclusion.split('\n')[0]
            reasoning_step = replace_conclusion(reasoning_step, revised_conclusion)
            print("\n==================Revised Step ==================")
            print(reasoning_step)
            print("================================================")
            cur_ex['proof'] += " " + reasoning_step
            print(prover_prompt.format(**cur_ex))

        print("\n==================Generated Proof==================")
        print(cur_ex['proof'])
        print("================================================")

        print("\n==================Ground Truth==================")
        print(example['proof'])
        print("================================================")

        input()


if __name__ == '__main__':
    Fire(run)