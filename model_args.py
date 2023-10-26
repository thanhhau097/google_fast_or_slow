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
