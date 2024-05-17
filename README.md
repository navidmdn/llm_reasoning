# llm_reasoning
A repo to understand the reasoning capabilities of llms


## Select and Deduct

The implemented method for this section is explained in [this blog post](https://navidmdn.github.io/homepage/blog/reasoning-select-and-deduct/). The general idea is to implement a simple, 
modular and explainable reasoning system that tries to mimic the reasoning capabilities of humans in a combination of
forward and backward reasoning. 

### Training the models yourself

#### Downloading Data

To prepare the dataset for training the models, download both `ProofWriter` and `EntailmentBank` datasets from [this link](https://drive.google.com/file/d/1AJE8LIH-V_vYSJ0iZI4I3KuRJqPObT_g/view?usp=sharing)
and extract both of the folders inside the `data` folder. The `ProofWriter` dataset is post-processed according to the
[NLProofs](https://github.com/princeton-nlp/NLProofS) repo to match the same format as the `EntailmentBank` dataset.

#### Preprocessing and Training

The `select_and_deduct/prepare_data_for_seq2seq_training.sh` script will preprocess the downloaded data to prepare augmented
datasets containing a balanced combination of both `ProofWriter` and `EntailmentBank` datasets from all subtasks. For `ProofWriter`
we take all correct proofs from all depths and sample it with the same number of examples as the `EntailmentBank` dataset.
For the `EntailmentBank` dataset, we take all the examples from subtask 1 and 2.

The script will use `select_and_deduct/deductor_data_handler.py` and `select_and_deduct/selector_data_handler.py` to prepare the data. 
By default, the deductor will append the final hypothesis to the input which you can control by `--add_hypothesis` flag. 
After preparing data, you will get 4 files in `data/` directory which are training and validation files for both the selector and the deductor.

To train the models, you can use the `select_and_deduct/run_train_deductor_flant5.sh` and `select_and_deduct/run_train_selector_flant5.sh` scripts 
which will train a flan-t5 model for both tasks.

#### Inference

You can run the proof search algorithm by running the `select_and_deduct/proof_search.py` script. Here are a few important arguments
to control the search algorithm:

```bash
PYTHONPATH=.. python proof_search.py \
  --deductor_path ../outputs/deductor-flant5-xl/ \
  --selector_path ../outputs/selector-flant5-xl/ \
  --test_data_path ../data/test/ent_task2.json \
  --output_dir ent_task2_hyp_th0.5_beamsearch/ \
  --n_search_beams 8 \
  --hypothesis_acceptance_threshold 0.5 \
  --search_algorithm beam_search \
  --deductor_add_hyp
```

You can add or remove hypothesis from the input of the deductor by using the `--deductor_add_hyp` flag with respect to the training configuration. Also,
you can control the number of search beams by using the `--n_search_beams` flag. The `--search_algorithm` flag can be set to `beam_search`, `greedy`
or `early_selection`. You can also set the deductor and selector paths to the trained models or the pretrained models we discuss in the
next section.

### Using the pretrained models

Both selector and deductor models are trained and shared on huggingface so that you can use them directly ([deductor model](https://huggingface.co/navidmadani/deductor_flant5_large_v1.0), [selector model](https://huggingface.co/navidmadani/selector_flant5_large_v1.0)). You can use the code
inside `select_and_deduct/inference.py` to run a gradio server and interact with the models with your custom inputs. It will
load the pretrained models and run the proof search algorithm to find the proofs for the given input and plots the entailment
graph along with intermediate steps as shown below:

![Gradio Interface](/select_and_deduct/statics/gradio-example.png)

