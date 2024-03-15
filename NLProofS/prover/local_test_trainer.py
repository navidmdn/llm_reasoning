from prover.datamodule import ProofDataModule, StepwiseDataset
from prover.model import EntailmentWriter
import os
from lightning.pytorch.trainer import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    t = Trainer(accelerator='mps', limit_train_batches=5, limit_val_batches=1, limit_test_batches=1)

    model = EntailmentWriter(
        dataset='entailmentbank',
        max_input_len=20,
        stepwise=True,
        max_num_steps=20,
        lr=0.0001,
        warmup_steps=2,
        model_name='t5-small',
        num_beams=10,
        topk=10,
        verifier_ckpt=None,
        verifier_weight=0.0,
        proof_search=False,
        oracle_prover=False,
        oracle_verifier=False
    )

    dm = ProofDataModule(
        dataset='entailmentbank',
        model_name='t5-small',
        stepwise=True,
        sample_goal='hypothesis',
        subtree_proved_prob=0.5,
        subtree_proved_all_or_none=True,
        batch_size=1,
        num_workers=1,
        max_input_len=1024,
        max_output_len=64,
        path_train='../data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl',
        path_val='../data/entailment_trees_emnlp2021_data_v3/dataset/task_1/dev.jsonl',
        path_test='../data/entailment_trees_emnlp2021_data_v3/dataset/task_1/test.jsonl'
    )

    # ds_train = StepwiseDataset(
    #     dm.dataset,
    #     dm.path_train,
    #     dm.model_name,
    #     dm.max_input_len,
    #     dm.max_output_len,
    #     dm.sample_goal,
    #     dm.subtree_proved_prob,
    #     dm.subtree_proved_all_or_none,
    #     is_train=True,
    # )
    # dm.ds_train = ds_train
    #
    # dl = dm.train_dataloader()
    # for batch in dl:
    #     print(batch)

    t.fit(model, dm)

