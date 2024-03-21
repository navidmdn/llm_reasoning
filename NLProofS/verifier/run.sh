PL_TORCH_DISTRIBUTED_BACKEND='nccl' WANDB_CACHE_DIR=../../../wandbcache \
  PYTHONPATH=.. WANDB_ENTITY=navidmdn \
  WANDB_PROJECT=nlproofs python main.py fit\
  --config cli_entailmentbank_task2.yaml