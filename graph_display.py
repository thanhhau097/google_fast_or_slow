import graphviz
import random
from pathlib import Path
import numpy as np


code_mapping = {
    1: "abs",
    2: "add",
    3: "add-dependency",
    4: "after-all",
    5: "all-reduce",
    6: "all-to-all",
    7: "atan2",
    8: "batch-norm-grad",
    9: "batch-norm-inference",
    10: "batch-norm-training",
    11: "bitcast",
    12: "bitcast-convert",
    13: "broadcast",
    14: "call",
    15: "ceil",
    16: "cholesky",
    17: "clamp",
    18: "collective-permute",
    19: "count-leading-zeros",
    20: "compare",
    21: "complex",
    22: "concatenate",
    23: "conditional",
    24: "constant",
    25: "convert",
    26: "convolution",
    27: "copy",
    28: "copy-done",
    29: "copy-start",
    30: "cosine",
    31: "custom-call",
    32: "divide",
    33: "domain",
    34: "dot",
    35: "dynamic-slice",
    36: "dynamic-update-slice",
    37: "exponential",
    38: "exponential-minus-one",
    39: "fft",
    40: "floor",
    41: "fusion",
    42: "gather",
    43: "get-dimension-size",
    44: "set-dimension-size",
    45: "get-tuple-element",
    46: "imag",
    47: "infeed",
    48: "iota",
    49: "is-finite",
    50: "log",
    51: "log-plus-one",
    52: "and",
    53: "not",
    54: "or",
    55: "xor",
    56: "map",
    57: "maximum",
    58: "minimum",
    59: "multiply",
    60: "negate",
    61: "outfeed",
    62: "pad",
    63: "parameter",
    64: "partition-id",
    65: "popcnt",
    66: "power",
    67: "real",
    68: "recv",
    69: "recv-done",
    70: "reduce",
    71: "reduce-precision",
    72: "reduce-window",
    73: "remainder",
    74: "replica-id",
    75: "reshape",
    76: "reverse",
    77: "rng",
    78: "rng-get-and-update-state",
    79: "rng-bit-generator",
    80: "round-nearest-afz",
    81: "rsqrt",
    82: "scatter",
    83: "select",
    84: "select-and-scatter",
    85: "send",
    86: "send-done",
    87: "shift-left",
    88: "shift-right-arithmetic",
    89: "shift-right-logical",
    90: "sign",
    91: "sine",
    92: "slice",
    93: "sort",
    94: "sqrt",
    95: "subtract",
    96: "tanh",
    97: "transpose",
    98: "triangular-solve",
    99: "tuple",
    100: "while",
    101: "cbrt",
    102: "all-gather",
    103: "collective-permute-start",
    104: "collective-permute-done",
    105: "logistic",
    106: "dynamic-reshape",
    107: "all-reduce-start",
    108: "all-reduce-done",
    109: "reduce-scatter",
    110: "all-gather-start",
    111: "all-gather-done",
    112: "opt-barrier",
    113: "async-start",
    114: "async-update",
    115: "async-done",
    116: "round-nearest-even",
    117: "stochastic-convert",
    118: "tan",
}


# NOTE: PNG is much slower and does not open on VSCode
# For SVG visualization install the `Svg Preview` package (not the `SVG` package)
def gen_graph_display(path, out_path, format="svg"):
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = dict(np.load(path, allow_pickle=True))

    dot = graphviz.Graph(engine='sfdp', format=format)
    dot.attr(overlap='scale')

    for i in range(len(data["node_feat"])):
        dot.node(str(i), code_mapping[data["node_opcode"][i]])

    for i, j in data["edge_index"]:
        dot.edge(str(int(i)), str(int(j)))

    dot.render(filename=out_path, view=False)


if __name__ == "__main__":
    gen_graph_display(
        "./data_new/npz/layout/xla/default/train/resnet50.2x2.fp16.npz",
        "./large_graph"
    )

    gen_graph_display(
        "./data_new/npz/layout/xla_pruned/default/train/resnet50.2x2.fp16.npz",
        "./large_graph_pruned"
    )


