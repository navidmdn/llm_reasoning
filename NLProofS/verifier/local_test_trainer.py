from verifier.datamodule import EntailmentDataModule
from verifier.model import EntailmentClassifier
import os
from lightning.pytorch.trainer import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    t = Trainer(accelerator='cpu', limit_train_batches=5, limit_val_batches=1, limit_test_batches=1)

    model = EntailmentClassifier(
        lr=0.0001,
        warmup_steps=2,
        model_name='roberta-base',
        pos_weight=128.0,
        max_input_len=256,
    )

    dm = EntailmentDataModule(
        dataset='entailmentbank',
        model_name='roberta-base',
        max_num_premises=4,
        batch_size=2,
        num_workers=1,
        irrelevant_distractors_only=False,
        max_input_len=256,
        path_train='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.jsonl',
        path_val='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl',
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

