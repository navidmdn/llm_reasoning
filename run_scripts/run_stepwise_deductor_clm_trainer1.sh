#python examples/scripts/sft.py \
#    --model_name_or_path= \
#    --report_to="wandb" \
#    --learning_rate=1.41e-5 \
#    --per_device_train_batch_size=64 \
#    --gradient_accumulation_steps=16 \
#    --output_dir="sft_openassistant-guanaco" \
#    --logging_steps=1 \
#    --num_train_epochs=3 \
#    --max_steps=-1 \
#    --push_to_hub \
#    --gradient_checkpointing \
#    --use_peft \
#    --lora_r=64 \
#    --lora_alpha=16

WANDB_MODE=offline WANDB_ENTITY=navidmdn WANDB_PROJECT=deductor-clm PYTHONPATH=.. python ../select_and_deduct/train_clm_sft.py\
  --train_file ../data/deductor_train_clm.json\
  --dev_file ../data/deductor_dev_clm.json\
  --output_dir ../outputs/test_clm/\
  --model_id meta-llama/Llama-2-7b-chat-hf\
  --cache_dir ../../hfcache/\
  --use_peft\
  --load_in_4bit
