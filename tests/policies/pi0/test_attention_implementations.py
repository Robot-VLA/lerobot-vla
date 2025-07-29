import time
from typing import Any, Dict

import pytest
import torch

from lerobot.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from tests.utils import DEVICE

BATCH_SIZE = 64
SEQ_LEN = 64


@pytest.fixture
def action_expert():
    """Fixture to provide the action expert model."""
    cfg = PaliGemmaWithExpertConfig(
        attention_implementation="eager",
        train_expert_only=True,
    )
    model = PaliGemmaWithExpertModel(cfg)
    model.eval()
    return model


def _create_test_inputs(
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    device: str = DEVICE,
) -> Dict[str, Any]:
    num_att_heads = 8
    num_kv_heads = 1
    head_dim = 256

    query_states = torch.randn(
        batch_size, seq_len, num_att_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key_states = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value_states = torch.randn(
        batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    attention_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=device, dtype=torch.bool))

    return {
        "attention_mask": attention_mask,
        "batch_size": batch_size,
        "head_dim": head_dim,
        "query_states": query_states,
        "key_states": key_states,
        "value_states": value_states,
    }


@pytest.fixture
def test_inputs():
    return _create_test_inputs()


def _timed_forward(interface, inputs):
    """Forward pass with CUDA‑sync timing; returns (output, ms)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = interface(**inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1_000  # milliseconds
    return out, ms


def test_attention_correctness(action_expert, test_inputs):
    timings = {}

    # 1. eager
    attn = action_expert.get_attention_interface()
    eager_out, timings["eager"] = _timed_forward(attn, test_inputs)

    # 2. flex
    action_expert.config.attention_implementation = "flex"
    attn = action_expert.get_attention_interface()
    flex_out, timings["flex"] = _timed_forward(attn, test_inputs)

    # 3. sdpa
    action_expert.config.attention_implementation = "sdpa"
    attn = action_expert.get_attention_interface()
    sdpa_out, timings["sdpa"] = _timed_forward(attn, test_inputs)

    # 4. compare outputs (relaxed tol)
    tol = 1e-1
    assert torch.allclose(eager_out, flex_out, atol=tol), "eager vs flex mismatch"
    assert torch.allclose(eager_out, sdpa_out, atol=tol), "eager vs sdpa mismatch"
    assert torch.allclose(flex_out, sdpa_out, atol=tol), "flex  vs sdpa mismatch"

    # 5. print timings
    for impl, ms in timings.items():
        print(f"{impl:5s}: {ms:8.2f} ms")
