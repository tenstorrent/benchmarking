#!/usr/bin/env python3

#
# Compare two lists of tensors dumped into pickle files. Mainly meant to be
# used for intermediate tensors dumped from benchmark.py with -di


import pickle
import sys

import pybuda
from pybuda.op.eval.common import compare_tensor_to_golden


def load_tensor(filename):
    try:
        with open(filename, "rb") as f:
            ret = pickle.load(f)
            for i, t in enumerate(ret):
                if isinstance(t, pybuda.tensor.TensorFromPytorch):
                    assert t.has_value()
                    ret[i] = t.value()
            return ret
    except Exception:
        return None


def main():
    if len(sys.argv) < 3:
        print("Usage: %s <dump1> <dump2>" % sys.argv[0])
        sys.exit(1)

    f1 = sys.argv[1]
    f2 = sys.argv[2]

    t1s = load_tensor(f1)
    t2s = load_tensor(f2)

    for i in range(len(t1s)):
        compare_tensor_to_golden(f"cmp{i}", t1s[i], t2s[i], pcc=0.99, relative_atol=0.2)


if __name__ == "__main__":
    main()
