"""
uv run pytest -s tests/policies/pi0/test_rope.py

Expected output:
torch_rope(cuda):    45.69 ms
jax_rope  (cuda):     2.42 ms
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from lerobot.policies.pi0.conversion_scripts.openpi.src.openpi.models.gemma import _apply_rope
from lerobot.policies.pi0.paligemma_with_expert import apply_rope
from tests.utils import DEVICE

BATCH, SEQ, HEADS, DIM = 64, 128, 8, 64


def _torch_sync():
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_torch(fn, *t):
    _torch_sync()
    t0 = time.perf_counter()
    out = fn(*t)
    _torch_sync()
    return out, (time.perf_counter() - t0) * 1_000


def _timed_jax(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    jax.block_until_ready(out)
    return out, (time.perf_counter() - t0) * 1_000


def _jax_device():
    if DEVICE.startswith("cuda"):
        return jax.devices("gpu")[0]
    if DEVICE.startswith("mps"):
        return jax.devices("cpu")[0]  # JAX has no MPS yet
    return jax.devices("cpu")[0]


@pytest.fixture
def tensors():
    torch_dev = torch.device(DEVICE)
    x_torch = torch.randn(BATCH, SEQ, HEADS, DIM, dtype=torch.bfloat16, device=torch_dev)
    pos_torch = torch.arange(SEQ, device=torch_dev).expand(BATCH, SEQ)

    x_jax = jax.device_put(jnp.asarray(x_torch.cpu().float().numpy(), dtype=jnp.bfloat16), _jax_device())
    pos_jax = jax.device_put(jnp.asarray(pos_torch.cpu().numpy(), dtype=jnp.int32), _jax_device())

    return x_torch, pos_torch, x_jax, pos_jax


def test_apply_rope(tensors):
    xt, pt, xj, pj = tensors

    yt, tt = _timed_torch(apply_rope, xt, pt)  # PyTorch
    _, _ = _timed_jax(_apply_rope, xj, positions=pj)  # warm‑up compile
    yj, tj = _timed_jax(_apply_rope, xj, positions=pj)  # JAX

    assert yt.shape == xt.shape == tuple(yj.shape)
    assert yt.dtype == xt.dtype
    torch.testing.assert_close(xt, _timed_torch(apply_rope, xt, torch.zeros_like(pt))[0], rtol=0, atol=0)
    assert jnp.allclose(xj, _timed_jax(_apply_rope, xj, positions=jnp.zeros_like(pj))[0])

    yj_torch = torch.from_numpy(np.asarray(yj, dtype=np.float32)).to(dtype=torch.float32).to(DEVICE)
    yt_fp32 = yt.to(dtype=torch.float32).to(DEVICE)
    torch.testing.assert_close(yt_fp32, yj_torch, rtol=1e-2, atol=1e-2)

    print()
    print(f"torch_rope({DEVICE}): {tt:8.2f} ms")
    print(f"jax_rope  ({DEVICE}): {tj:8.2f} ms")
