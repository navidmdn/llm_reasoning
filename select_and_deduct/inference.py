from proof_search import proof_beam_search, load_model_and_tokenizer, load_examples
from visualize_entailment_tree import visualize_from_proof, visualize_tree
from fire import Fire
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import networkx as nx
import argparse
import gradio as gr
import matplotlib.pyplot as plt
import json


def run_inference(premises, hypothesis, deductor, deductor_tokenizer, selector, selector_tokenizer,
                  bleurt_tokenizer, bleurt_model, n_search_beams=5, hypothesis_acceptance_threshold=0.5,
                  verbose=False, deductor_add_hyp=False):
    premises_str = " ".join([f"{k}: {v}" for k, v in premises.items()])
    example = {'context': premises_str, 'hypothesis': hypothesis}
    proof = proof_beam_search(example, deductor, deductor_tokenizer, selector, selector_tokenizer,
                              bleurt_tokenizer, bleurt_model, n_search_beams=n_search_beams,
                              hypothesis_acceptance_threshold=hypothesis_acceptance_threshold,
                              verbose=verbose, deductor_add_hyp=deductor_add_hyp)

    try:
        print(visualize_tree(proof, example))
    except Exception as e:
        print(f"Failed to visualize tree: {e}")

    return proof

def plot_tree(tree):
    pos = nx.spring_layout(tree)
    labels = nx.get_node_attributes(tree, 'label')
    colors = [tree.nodes[node].get('color', 'skyblue') for node in tree.nodes]

    plt.figure(figsize=(8, 6))
    nx.draw(tree, pos, labels=labels, node_color=colors, with_labels=True, node_size=3000, font_size=10,
            font_color='black', font_weight='bold')
    plt.savefig('tree.png')
    return 'tree.png'


def generate_tree(premises, hypothesis):
    premises = [s.strip() for s in premises.split('\n') if len(s.strip()) > 0]
    premises = {f'sent{i + 1}': p for i, p in enumerate(premises)}
    hypothesis = hypothesis.strip()
    proof = run_inference(premises, hypothesis, deductor, deductor_tokenizer, selector,
                          selector_tokenizer,
                          bleurt_tokenizer, bleurt_model, n_search_beams=5, hypothesis_acceptance_threshold=0.5,
                          verbose=False, deductor_add_hyp=False)

    tree, updated_context = visualize_from_proof(premises, proof)
    tree_img = plot_tree(tree)
    proof_nodes = []
    for node in tree.nodes:
        if node in updated_context:
            proof_nodes.append(f"{node}: {updated_context[node]}")
    proof_nodes_str = '\n'.join(proof_nodes)
    return tree_img, proof_nodes_str


if __name__ == '__main__':
    # add argparse to specify the model names and cache_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--deductor_model_name', type=str, default="navidmadani/deductor_flant5_large_v1.0",
                        help="The name of the deductor model to use.")
    parser.add_argument('--selector_model_name', type=str, default="navidmadani/selector_flant5_large_v1.0",
                        help="The name of the selector model to use.")
    parser.add_argument('--cache_dir', type=str, default=None,
                        help="The directory to cache the models and tokenizers.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deductor, deductor_tokenizer = load_model_and_tokenizer(args.deductor_model_name, cache_dir=args.cache_dir)
    deductor = deductor.to(device)
    deductor.eval()
    selector, selector_tokenizer = load_model_and_tokenizer(args.selector_model_name, cache_dir=args.cache_dir)
    selector.eval()
    selector = selector.to(device)

    bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512", cache_dir=args.cache_dir)
    bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512",
                                                                      cache_dir=args.cache_dir)
    bleurt_model.eval()
    bleurt_model = bleurt_model.to(device)

    interface = gr.Interface(
        fn=generate_tree,
        inputs=[
            gr.Textbox(lines=10, placeholder="Enter premises, one per line..."),
            gr.Textbox(lines=1, placeholder="Enter hypothesis...")
        ],
        outputs=[gr.Image(type="filepath"), gr.Textbox(lines=10)],
        title="Entailment Tree Builder",
        description="Generate an entailment tree based on the provided premises and hypothesis."
    )

    interface.launch()
