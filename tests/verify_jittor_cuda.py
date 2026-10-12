import ctypes
import math
import os

import jittor as jt
import numpy as np


def main():
    if int(np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError(f"Jittor 1.3.11 requires NumPy 1.x, found {np.__version__}")

    source = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    transferred = jt.array(source).numpy()
    if not np.array_equal(source, transferred):
        raise RuntimeError(f"NumPy/Jittor bridge mismatch: {transferred.tolist()}")
    if not np.array_equal(jt.zeros((2, 2)).numpy(), np.zeros((2, 2), dtype=np.float32)):
        raise RuntimeError("Jittor zero initialization failed")

    jt.flags.use_cuda = 1
    x = jt.ones((1, 3, 8, 8))
    conv = jt.nn.Conv(3, 4, 1)
    conv.weight.assign(jt.ones(conv.weight.shape)).sync()
    conv.bias.assign(jt.array(np.zeros(tuple(conv.bias.shape), dtype=np.float32))).sync()
    y = conv(x).mean() + (x + 1).sum()
    y.sync()
    value = float(y.item())
    expected = 387.0
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError(f"CUDA result mismatch: {value} != {expected}")
    cudnn = ctypes.CDLL("libcudnn.so")
    version = int(cudnn.cudnnGetVersion())
    print("jittor", jt.__version__)
    print("numpy", np.__version__)
    print("cuda", os.environ.get("nvcc_path", "auto"))
    print("cudnn", version)
    print("gpu", int(jt.flags.use_cuda))
    print("jittor_cuda_passed", value)


if __name__ == "__main__":
    main()
