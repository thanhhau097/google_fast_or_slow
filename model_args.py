from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments relating to model.
    """

    resume: Optional[str] = field(default=None, metadata={"help": "Path of model checkpoint"})
    hidden_channels: str = field(
        default="32,48,64,84",
        metadata={"help": "Hidden channels for graph convolutions"},
    )
    graph_in: int = field(default=256, metadata={"help": "input graph embedding size"})
    graph_out: int = field(default=256, metadata={"help": "output graph embedding size"})
    hidden_dim: int = field(default=256, metadata={"help": "hidden dimension"})
    dropout: float = field(default=0.2, metadata={"help": "dropout rate"})
    gat_dropout: float = field(default=0.2, metadata={"help": "dropout rate at gcn level"})
    op_embedding_dim: int = field(default=16, metadata={"help": "num of op embedding dim"})
    layout_embedding_dim: int = field(default=4, metadata={"help": "num of layout embedding dim"})
    norm: str = field(default="instance", metadata={"help": "instance norm or group norm"})
    use_cross_attn: bool = field(default=False, metadata={"help": "use cross attn or not"})
    weights_folder: str = field(
        default="weights", metadata={"help": "folder to load weights for prediction"}
    )
