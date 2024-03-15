from common import *
from lightning.pytorch.cli import LightningCLI
from verifier.datamodule import EntailmentDataModule
from verifier.model import EntailmentClassifier
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: Any) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_input_len", "model.max_input_len")


def main() -> None:
    #  temporary fix for slurm issue
    for var in os.environ:
        if 'slurm' in var.lower():
            print(var)
            del os.environ[var]

    cli = CLI(EntailmentClassifier, EntailmentDataModule, save_config_kwargs={"overwrite": True})
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
