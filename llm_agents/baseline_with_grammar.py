import os
from operator import itemgetter

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

with open('./secret', 'r') as f:
    secret = f.read().strip()

os.environ['LANGCHAIN_API_KEY'] = secret



from llm_agents.utils import load_ollama_autoregressive_model, load_llamacpp_autoregressive_model
from llm_agents.data_handler import load_entailemnt_tree_dataset, get_processed_entailmenet_dataset
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotPromptTemplate
from fire import Fire
from langchain.prompts.example_selector import NGramOverlapExampleSelector, SemanticSimilarityExampleSelector


def run(llm_name='mistral:instruct', cache_dir=None):

    llm = load_llamacpp_autoregressive_model(model_file=llm_name)

    ## load data and few shot examples ##
    train_examples = load_entailemnt_tree_dataset('./data/entailment_trees/train-task1.jsonl')
    valid_examples = load_entailemnt_tree_dataset('./data/entailment_trees/train-task1.jsonl')
    train_examples, valid_examples = get_processed_entailmenet_dataset(train_examples, valid_examples)

    example_prompt = PromptTemplate(
        input_variables=["context", "hypothesis", "proof"],
        template="premises:\n{context}\nhypothesis:\n{hypothesis}\nproof:\n{proof}\n"
    )

    top_k = 5
    example_selector = NGramOverlapExampleSelector(
        examples=train_examples[:top_k],
        example_prompt=example_prompt,
    )

    for example in valid_examples:
        # selected_examples = example_selector.select_examples(example)
        # print(f"Examples most similar to the input {example} are: {selected_examples}")

        prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="""You are a reasoning and logical prover. Your task is to generate a natural language proof to reach\
 a hypothesis. Here are a few exapmles and the format you need to follow to generate the proof.""",
            suffix="premises:\n{context}\nhypothesis:\n{hypothesis}\nproof:\n""",
            input_variables=["context", "hypothesis"],
        )

        print(prompt.format(**example))

        gen_chain = prompt | llm | StrOutputParser()

        for chunk in gen_chain.stream(example):
            print(chunk, end="", flush=True)

        print("\n==================Ground Truth==================")
        print(example['proof'])
        print("================================================")

        break


if __name__ == '__main__':
    Fire(run)