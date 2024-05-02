
WANDB_MODE=online WANDB_ENTITY=navidmdn WANDB_PROJECT=selector-clm PYTHONPATH=.. python train_clm_sft.py\
  --train_file ../data/selector_train_clm.json\
  --dev_file ../data/selector_dev_clm.json\
  --output_dir ../outputs/selector_llama2-7b\
  --do_train\
  --do_eval\
  --model_id meta-llama/Llama-2-7b-hf\
  --cache_dir ../../hfcache/\
  --use_peft\
  --load_in_4bit\
  --evaluation_strategy steps\
  --include_inputs_for_metrics\
  --per_device_train_batch_size 4\
  --per_device_eval_batch_size 4\
  --gradient_accumulation_steps 8\
  --eval_accumulation_steps 1\
  --num_train_epochs 10\
  --save_strategy steps\
  --save_total_limit 1\
  --metric_for_best_model eval_top5_acc\
  --evaluation_strategy steps\
  --save_steps 10\
  --logging_steps 5\
  --overwrite_output_dir\
  --report_to wandb\
  --load_best_model_at_end\
  --eval_steps 10\
  --max_eval_samples 128\
  --max_new_tokens 20\
  --num_return_sequences 5\
  --max_seq_length 1024\
  --flash_attention\




