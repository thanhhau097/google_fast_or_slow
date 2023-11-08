import graphviz
import random
from pathlib import Path
import numpy as np


code_mapping = {
    "abs": 1,
    "add": 2,
    "add-dependency": 3,
    "after-all": 4,
    "all-reduce": 5,
    "all-to-all": 6,
    "atan2": 7,
    "batch-norm-grad": 8,
    "batch-norm-inference": 9,
    "batch-norm-training": 10,
    "bitcast": 11,
    "bitcast-convert": 12,
    "broadcast": 13,
    "call": 14,
    "ceil": 15,
    "cholesky": 16,
    "clamp": 17,
    "collective-permute": 18,
    "count-leading-zeros": 19,
    "compare": 20,
    "complex": 21,
    "concatenate": 22,
    "conditional": 23,
    "constant": 24,
    "convert": 25,
    "convolution": 26,
    "copy": 27,
    "copy-done": 28,
    "copy-start": 29,
    "cosine": 30,
    "custom-call": 31,
    "divide": 32,
    "domain": 33,
    "dot": 34,
    "dynamic-slice": 35,
    "dynamic-update-slice": 36,
    "exponential": 37,
    "exponential-minus-one": 38,
    "fft": 39,
    "floor": 40,
    "fusion": 41,
    "gather": 42,
    "get-dimension-size": 43,
    "set-dimension-size": 44,
    "get-tuple-element": 45,
    "imag": 46,
    "infeed": 47,
    "iota": 48,
    "is-finite": 49,
    "log": 50,
    "log-plus-one": 51,
    "and": 52,
    "not": 53,
    "or": 54,
    "xor": 55,
    "map": 56,
    "maximum": 57,
    "minimum": 58,
    "multiply": 59,
    "negate": 60,
    "outfeed": 61,
    "pad": 62,
    "parameter": 63,
    "partition-id": 64,
    "popcnt": 65,
    "power": 66,
    "real": 67,
    "recv": 68,
    "recv-done": 69,
    "reduce": 70,
    "reduce-precision": 71,
    "reduce-window": 72,
    "remainder": 73,
    "replica-id": 74,
    "reshape": 75,
    "reverse": 76,
    "rng": 77,
    "rng-get-and-update-state": 78,
    "rng-bit-generator": 79,
    "round-nearest-afz": 80,
    "rsqrt": 81,
    "scatter": 82,
    "select": 83,
    "select-and-scatter": 84,
    "send": 85,
    "send-done": 86,
    "shift-left": 87,
    "shift-right-arithmetic": 88,
    "shift-right-logical": 89,
    "sign": 90,
    "sine": 91,
    "slice": 92,
    "sort": 93,
    "sqrt": 94,
    "subtract": 95,
    "tanh": 96,
    "trace": 97,
    "transpose": 98,
    "triangular-solve": 99,
    "tuple": 100,
    "tuple-select": 101,
    "while": 102,
    "cbrt": 103,
    "all-gather": 104,
    "collective-permute-start": 105,
    "collective-permute-done": 106,
    "logistic": 107,
    "dynamic-reshape": 108,
    "all-reduce-start": 109,
    "all-reduce-done": 110,
    "reduce-scatter": 111,
    "all-gather-start": 112,
    "all-gather-done": 113,
    "opt-barrier": 114,
    "async-start": 115,
    "async-update": 116,
    "async-done": 117,
    "round-nearest-even": 118,
    "stochastic-convert": 119,
    "tan": 120,
}

inv_code_mapping = {v: k for k, v in code_mapping.items()}


# NOTE: PNG is much slower and does not open on VSCode
# For SVG visualization install the `Svg Preview` package (not the `SVG` package)
def gen_graph_display(path, out_path, format="svg"):
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = dict(np.load(path, allow_pickle=True))

    dot = graphviz.Graph(engine="sfdp", format=format)
    dot.attr(overlap="scale")

    print(len(data["node_feat"]))

    for i in range(len(data["node_feat"])):
        dot.node(str(i), inv_code_mapping[data["node_opcode"][i]])

    for i, j in data["edge_index"]:
        dot.edge(str(int(i)), str(int(j)))

    dot.render(filename=out_path, view=False)


if __name__ == "__main__":
    gen_graph_display(
        "./data_new/npz/layout/xla/default/train/resnet50.2x2.fp16.npz", "./large_graph"
    )

    gen_graph_display(
        "./data_new/npz/layout/xla_pruned/default/train/resnet50.2x2.fp16.npz",
        "./large_graph_pruned",
    )
