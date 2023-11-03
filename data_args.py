from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    data_folder: str = field(
        default="data/npz_all/npz/",
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
