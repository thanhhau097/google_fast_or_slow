from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    data_folder: str = field(
        default="data/npz_pad",
        metadata={"help": "The folder containing the data files."},
    )
    data_type: str = field(
        default="tile", metadata={"help": "The type of data to use: layout/tile"}
    )
    source: str = field(default="xla", metadata={"help": "'xla' or 'nlp'"})
    search: str = field(default="default", metadata={"help": "'default' or 'random' or 'mix"})
    data_concatenation: bool = field(
        default=False, metadata={"help": "Whether to concatenate data from different searches"}
    )
    use_compressed: bool = field(
        default=True, metadata={"help": "Whether to use compressed data"}
    )
    max_configs: int = field(default=128, metadata={"help": "max number of configs per graph"})
    max_configs_eval: int = field(
        default=512, metadata={"help": "max number of configs per graph for validation"}
    )
    select_close_runtimes: bool = field(
        default=False, metadata={"help": "Whether to select close runtimes"}
    )
    select_close_runtimes_prob: float = field(
        default=0.5, metadata={"help": "Probability of selecting close runtimes"}
    )
    filter_random_configs: bool = field(
        default=False, metadata={"help": "Whether to filter random configs when training with mix data"}
    )
    use_standard_scaler: bool = field(
        default=False, metadata={"help": "Whether to use standard scaler"}
    )
    fold: int = field(
        default=0, metadata={"help": "Which fold to use. 0-8 for kfold, -1 for all folds"}
    )

    # finetuning base on architecture
    architecture_finetune: bool = field(
        default=False, metadata={"help": "Whether to finetune architecture"}
    )
    architecture_finetune_epochs: int = field(
        default=100, metadata={"help": "Number of epochs to finetune architecture"}
    )
    architecture_finetune_eval_steps: int = field(
        default=1000, metadata={"help": "Number of steps to evaluate architecture"}
    )
    architecture_finetune_test_file_names: str = field(
        default="all", metadata={"help": "The name of the test files, or 'all' for all test files"}
    )
