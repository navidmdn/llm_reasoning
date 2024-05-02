
WANDB_MODE=online WANDB_ENTITY=navidmdn WANDB_PROJECT=deductor-clm PYTHONPATH=.. python train_clm_sft.py\
  --train_file ../data/deductor_train_clm.json\
  --dev_file ../data/deductor_dev_clm.json\
  --output_dir ../outputs/deductor_llama2-7b_nohyp\
  --do_train\
  --do_eval\
  --model_id meta-llama/Llama-2-7b-chat-hf\
  --cache_dir ../../hfcache/\
  --use_peft\
  --load_in_4bit\
  --include_inputs_for_metrics\
  --per_device_train_batch_size 8\
  --per_device_eval_batch_size 8\
  --gradient_accumulation_steps 4\
  --num_train_epochs 10\
  --save_strategy steps\
  --evaluation_strategy steps\
  --save_total_limit 1\
  --metric_for_best_model eval_loss\
  --evaluation_strategy steps\
  --save_steps 10\
  --logging_steps 5\
  --overwrite_output_dir\
  --report_to wandb\
  --load_best_model_at_end\
  --eval_steps 10\
  --max_eval_samples 128\
  --max_sequnece_length 2048\



