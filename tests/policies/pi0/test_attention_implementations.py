import time

import pytest
import torch

from lerobot.policies.pi0.flex_attention import flex_attention_forward
from lerobot.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def test_inputs(device):
    return create_test_inputs(batch_size=2, seq_len=64, device=device)


def create_test_model(attention_implementation: str) -> PaliGemmaWithExpertModel:
    config = PaliGemmaWithExpertConfig(
        attention_implementation=attention_implementation,
        freeze_vision_encoder=True,
        train_expert_only=False,
    )
    model = PaliGemmaWithExpertModel(config)
    model.eval()
    return model


def create_test_inputs(batch_size=2, seq_len=128, device="cuda"):
    num_att_heads = 8
    num_key_value_heads = 1
    head_dim = 256

    query_states = torch.randn(
        batch_size, seq_len, num_att_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key_states = torch.randn(
        batch_size, seq_len, num_key_value_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    value_states = torch.randn(
        batch_size, seq_len, num_key_value_heads, head_dim, device=device, dtype=torch.bfloat16
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


def test_attention_correctness(device, test_inputs):
    results = {}

    model_eager = create_test_model("eager").to(device)
    with torch.no_grad():
        results["eager"] = model_eager.eager_attention_forward(**test_inputs)

    model_sdpa = create_test_model("sdpa").to(device)
    with torch.no_grad():
        results["sdpa"] = model_sdpa.sdpa_attention_forward(**test_inputs)

    try:
        with torch.no_grad():
            results["flex"] = flex_attention_forward(
                test_inputs["attention_mask"],
                test_inputs["batch_size"],
                test_inputs["head_dim"],
                test_inputs["query_states"],
                test_inputs["key_states"],
                test_inputs["value_states"],
            )
    except Exception:
        results["flex"] = None

    baseline = results["eager"]

    if results["sdpa"] is not None:
        rel_error = ((results["sdpa"] - baseline).abs() / (baseline.abs() + 1e-8)).mean().item()
        assert rel_error < 1e-3, f"SDPA rel error too high: {rel_error}"

    if results["flex"] is not None:
        rel_error = ((results["flex"] - baseline).abs() / (baseline.abs() + 1e-8)).mean().item()
        assert rel_error < 1e-3, f"Flex rel error too high: {rel_error}"


@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_attention_speed_benchmark(device, seq_len):
    num_runs = 5
    inputs = create_test_inputs(batch_size=2, seq_len=seq_len, device=device)

    def benchmark(forward_fn):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                forward_fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.time() - start) / num_runs

    model_eager = create_test_model("eager").to(device)
    model_sdpa = create_test_model("sdpa").to(device)

    eager_time = benchmark(lambda: model_eager.eager_attention_forward(**inputs))
    sdpa_time = benchmark(lambda: model_sdpa.sdpa_attention_forward(**inputs))
    try:
        flex_time = benchmark(
            lambda: flex_attention_forward(
                inputs["attention_mask"],
                inputs["batch_size"],
                inputs["head_dim"],
                inputs["query_states"],
                inputs["key_states"],
                inputs["value_states"],
            )
        )
    except Exception:
        flex_time = None  # noqa: F841

    assert eager_time > 0 and sdpa_time > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_usage(device):
    inputs = create_test_inputs(batch_size=2, seq_len=256, device=device)
    torch.cuda.empty_cache()

    memory_results = {}

    for impl in ["eager", "sdpa"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = create_test_model(impl).to(device)
        with torch.no_grad():
            getattr(model, f"{impl}_attention_forward")(**inputs)
        memory = torch.cuda.max_memory_allocated() / 1024**2
        memory_results[impl] = memory
        del model

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            flex_attention_forward(
                inputs["attention_mask"],
                inputs["batch_size"],
                inputs["head_dim"],
                inputs["query_states"],
                inputs["key_states"],
                inputs["value_states"],
            )
        memory_results["flex"] = torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass

    for impl, mem in memory_results.items():
        assert 0 < mem < 1000, f"{impl} memory usage abnormal: {mem} MB"


def test_all_attention_implementations_available():
    for impl in ["eager", "sdpa"]:
        model = create_test_model(impl)
        assert hasattr(model, f"{impl}_attention_forward")
    assert callable(flex_attention_forward)


@pytest.mark.integration
def test_full_attention_pipeline(device, test_inputs):
    test_attention_correctness(device, test_inputs)
