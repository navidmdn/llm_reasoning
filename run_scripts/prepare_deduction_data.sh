python ../select_and_deduct/stepwise_proofs_data_handler.py --input_path ../data/proofwriter-dataset-V2020.12.3/preprocessed_OWA\
  --output_path ../data/proofwriter-stepwise_proofs\
  --dataset ruletaker --add_hypothesis

python ../select_and_deduct/stepwise_proofs_data_handler.py --input_path ../data/entailment_trees_emnlp2021_data_v3\
  --output_path ../data/entailmenttree-stepwise_proofs\
  --dataset entailmenttree --add_hypothesis

python ../select_and_deduct/stepwise_proofs_data_handler.py --merge ../data/proofwriter-stepwise_proofs ../data/entailmenttree-stepwise_proofs\
  --output_path ../data/deductor_train_merged.json\
  --merge-equal --merge-equal --split train

python ../select_and_deduct/stepwise_proofs_data_handler.py --merge ../data/proofwriter-stepwise_proofs ../data/entailmenttree-stepwise_proofs\
  --output_path ../data/deductor_dev_merged.json\
  --merge-equal --merge-equal --split dev
