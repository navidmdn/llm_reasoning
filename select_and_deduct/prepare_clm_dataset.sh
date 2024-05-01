python prepare_clm_dataset.py\
  --input_file ../data/deductor_train_merged.json\
  --output_file ../data/deductor_train_clm.json\
  --task deduction

python prepare_clm_dataset.py\
  --input_file ../data/deductor_dev_merged.json\
  --output_file ../data/deductor_dev_clm.json\
  --task deduction

python prepare_clm_dataset.py\
  --input_file ../data/selector_train_merged.json\
  --output_file ../data/selector_train_clm.json\
  --task selection

python prepare_clm_dataset.py\
  --input_file ../data/selector_dev_merged.json\
  --output_file ../data/selector_dev_clm.json\
  --task selection

