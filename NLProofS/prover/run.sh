PL_TORCH_DISTRIBUTED_BACKEND='nccl' WANDB_CACHE_DIR=../../../wandbcache \
  PYTHONPATH=.. WANDB_ENTITY=navidmdn \
  WANDB_PROJECT=nlproofs-prover python main.py fit\
  --config cli_task2_stepwise_t5-large.yaml