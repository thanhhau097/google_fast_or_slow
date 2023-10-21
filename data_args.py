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
    search: str = field(default="default", metadata={"help": "'default' or 'random'"})
