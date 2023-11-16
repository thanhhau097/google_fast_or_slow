from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    data_folder: str = field(
        default="data/npz_all/npz",
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
