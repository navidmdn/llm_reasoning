python deductor_data_handler.py --input_path ../data/proofwriter-dataset-V2020.12.3/preprocessed_OWA\
  --output_path ../data/proofwriter-stepwise_proofs\
  --dataset ruletaker --add_hypothesis

python deductor_data_handler.py --input_path ../data/entailment_trees_emnlp2021_data_v3\
  --output_path ../data/entailmenttree-stepwise_proofs\
  --dataset entailmenttree --add_hypothesis

python deductor_data_handler.py --merge ../data/proofwriter-stepwise_proofs ../data/entailmenttree-stepwise_proofs\
  --output_path ../data/deductor_train_merged.json\
  --merge-equal --merge-equal --split train

python deductor_data_handler.py --merge ../data/proofwriter-stepwise_proofs ../data/entailmenttree-stepwise_proofs\
  --output_path ../data/deductor_dev_merged.json\
  --merge-equal --merge-equal --split dev

python selector_data_handler.py --input_path ../data/proofwriter-dataset-V2020.12.3/preprocessed_OWA\
  --output_path ../data/proofwriter-selector\
  --dataset ruletaker

python selector_data_handler.py --input_path ../data/entailment_trees_emnlp2021_data_v3\
  --output_path ../data/entailmenttree-selector\
  --dataset entailmenttree

python selector_data_handler.py --merge ../data/proofwriter-selector ../data/entailmenttree-selector\
  --output_path ../data/selector_train_merged.json\
  --merge-equal --merge-equal

python selector_data_handler.py --merge ../data/proofwriter-selector ../data/entailmenttree-selector\
  --output_path ../data/selector_dev_merged.json\
  --merge-equal --merge-equal