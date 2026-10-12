import math
import os
from typing import Iterable, Sequence

import jittor as jt
import numpy as np
from PIL import Image


def enable_cuda():
    if jt.has_cuda:
        jt.flags.use_cuda = 1


def _as_tuple(size):
    if isinstance(size, int):
        return (size, size)
    return tuple(size)


def read_image(path):
    image = Image.open(path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    array = np.array(image)
    if array.ndim == 2:
        array = array[None, :, :]
    else:
        array = array.transpose(2, 0, 1)
    return jt.array(array).uint8()


def _make_grid(tensor, nrow=8):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    n, c, h, w = tensor.shape
    nrow = max(1, min(nrow, n))
    ncol = int(math.ceil(n / nrow))
    grid = jt.zeros((c, ncol * h, nrow * w), dtype=tensor.dtype)
    for idx in range(n):
        row = idx // nrow
        col = idx % nrow
        grid[:, row * h:(row + 1) * h, col * w:(col + 1) * w] = tensor[idx]
    return grid


def save_image(tensor, path, nrow=8):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tensor = tensor.detach()
    if tensor.ndim == 4:
        tensor = _make_grid(tensor, nrow=nrow)
    tensor = jt.clamp(tensor, 0, 1)
    array = (tensor.numpy() * 255.0 + 0.5).astype(np.uint8)
    if array.shape[0] == 1:
        image = Image.fromarray(array[0], mode="L")
    else:
        image = Image.fromarray(array[:3].transpose(1, 2, 0), mode="RGB")
    image.save(path)


def interpolate(x, size, mode="bilinear"):
    size = _as_tuple(size)
    return jt.nn.interpolate(x, size=size, mode=mode)


def repeat_interleave(x, repeats):
    return jt.concat([x] * repeats, dim=0)


def chunk(x, chunks, dim=0):
    shape = x.shape[dim]
    step = shape // chunks
    parts = []
    for idx in range(chunks):
        start = idx * step
        end = shape if idx == chunks - 1 else (idx + 1) * step
        slices = [slice(None)] * x.ndim
        slices[dim] = slice(start, end)
        parts.append(x[tuple(slices)])
    return tuple(parts)


def rearrange(tensor, pattern, **axes):
    if pattern == "(b h) n d -> h (b n) d":
        h = axes["h"]
        bh, n, d = tensor.shape
        b = bh // h
        return tensor.reshape(b, h, n, d).permute(1, 0, 2, 3).reshape(h, b * n, d)
    if pattern == "h (b n) d -> b n (h d)":
        b = axes["b"]
        h, bn, d = tensor.shape
        n = bn // b
        return tensor.reshape(h, b, n, d).permute(1, 2, 0, 3).reshape(b, n, h * d)
    if pattern == "(h1 h) (b n) d -> (h1 b) n (h d)":
        b = axes["b"]
        h = axes["h"]
        h1h, bn, d = tensor.shape
        h1 = h1h // h
        n = bn // b
        return tensor.reshape(h1, h, b, n, d).permute(0, 2, 3, 1, 4).reshape(h1 * b, n, h * d)
    if pattern == "(b h) n d -> b n (h d)":
        h = axes["h"]
        bh, n, d = tensor.shape
        b = bh // h
        return tensor.reshape(b, h, n, d).permute(0, 2, 1, 3).reshape(b, n, h * d)
    if pattern == "b n (h d) -> (b h) n d":
        h = axes["h"]
        b, n, hd = tensor.shape
        d = hd // h
        return tensor.reshape(b, n, h, d).permute(0, 2, 1, 3).reshape(b * h, n, d)
    if pattern == "b ... -> b (...)":
        b = tensor.shape[0]
        return tensor.reshape(b, -1)
    if pattern == "b j -> (b h) () j":
        h = axes["h"]
        b, j = tensor.shape
        return tensor.unsqueeze(1).repeat(1, h, 1).reshape(b * h, 1, j)
    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")


def repeat(tensor, pattern, **axes):
    if pattern == "b j -> (b h) () j":
        return rearrange(tensor, pattern, **axes)
    raise NotImplementedError(f"Unsupported repeat pattern: {pattern}")


def set_seed(seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)
