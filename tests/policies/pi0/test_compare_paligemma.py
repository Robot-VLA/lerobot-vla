"""
uv run pytest -sv tests/policies/pi0/test_compare_paligemma.py

This test compares PaliGemmaForConditionalGeneration and GemmaForCausalLM models to ensure the
custom version is consistent with the original Huggingface implementation.

transformers==4.50.3

"""

import pytest
import torch
from transformers import (
    GemmaForCausalLM as HFGemmaForCausalLM,
)
from transformers import (
    PaliGemmaForConditionalGeneration as HFPaliGemmaForConditionalGeneration,
)
from transformers.models.auto import CONFIG_MAPPING

from lerobot.policies.pi0.models.gemma import GemmaConfig
from lerobot.policies.pi0.models.main import (
    GemmaForCausalLM,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
)
from tests.utils import DEVICE


@pytest.fixture(scope="module")
def hf_paligemma_model():
    paligemma_config = CONFIG_MAPPING["paligemma"](
        transformers_version="4.48.1",
        _vocab_size=257152,
        bos_token_id=2,
        eos_token_id=1,
        hidden_size=2048,
        image_token_index=257152,
        model_type="paligemma",
        pad_token_id=0,
        projection_dim=2048,
        text_config={
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "model_type": "gemma",
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "num_image_tokens": 256,
            "num_key_value_heads": 1,
            "torch_dtype": "float32",
            "vocab_size": 257152,
        },
        vision_config={
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "num_image_tokens": 256,
            "patch_size": 14,
            "projection_dim": 2048,
            "projector_hidden_act": "gelu_fast",
            "torch_dtype": "float32",
            "vision_use_head": False,
        },
    )
    paligemma = HFPaliGemmaForConditionalGeneration(paligemma_config)
    return paligemma.to(DEVICE)


@pytest.fixture(scope="module")
def hf_gemma_model():
    gemma_expert_config = CONFIG_MAPPING["gemma"](
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        hidden_activation="gelu_pytorch_tanh",
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type="gemma",
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype="float32",
        transformers_version="4.48.1",
        use_cache=True,
        vocab_size=257152,
    )
    gemma_expert = HFGemmaForCausalLM(config=gemma_expert_config)
    return gemma_expert.to(DEVICE)


@pytest.fixture(scope="module")
def paligemma_model(hf_paligemma_model):
    model = PaliGemmaForConditionalGeneration(PaliGemmaConfig()).to(DEVICE)
    model.load_state_dict(hf_paligemma_model.state_dict(), strict=False)
    return model


@pytest.fixture(scope="module")
def gemma_model(hf_gemma_model):
    model = GemmaForCausalLM(GemmaConfig()).to(DEVICE)
    model.load_state_dict(hf_gemma_model.state_dict(), strict=False)
    return model


def test_compare_paligemma_and_gemma(
    hf_paligemma_model: HFPaliGemmaForConditionalGeneration,
    hf_gemma_model: HFGemmaForCausalLM,
    paligemma_model: PaliGemmaForConditionalGeneration,
    gemma_model: GemmaForCausalLM,
):
    """
    Compare the Huggingface PaliGemmaForConditionalGeneration and GemmaForCausalLM models with custom implementations.
    """
    assert isinstance(hf_paligemma_model, HFPaliGemmaForConditionalGeneration)
    assert isinstance(hf_gemma_model, HFGemmaForCausalLM)

    assert isinstance(paligemma_model, PaliGemmaForConditionalGeneration)
    assert isinstance(gemma_model, GemmaForCausalLM)

    hf_paligemma_state_dict = hf_paligemma_model.state_dict()
    paligemma_state_dict = paligemma_model.state_dict()

    hf_gemma_state_dict = hf_gemma_model.state_dict()
    gemma_state_dict = gemma_model.state_dict()

    assert paligemma_state_dict.keys() <= hf_paligemma_state_dict.keys(), (
        "Custom PaliGemma model has keys not present in the HF version"
    )

    assert gemma_state_dict.keys() <= hf_gemma_state_dict.keys(), (
        "Custom Gemma model has keys not present in the HF version"
    )


def test_vision_tower(
    hf_paligemma_model: HFPaliGemmaForConditionalGeneration,
    hf_gemma_model: HFGemmaForCausalLM,
    paligemma_model: PaliGemmaForConditionalGeneration,
    gemma_model: GemmaForCausalLM,
):
    pixel_values = torch.randn(1, 3, 224, 224).to(DEVICE)
    hf_paligemma_outputs = hf_paligemma_model.vision_tower(pixel_values=pixel_values)
    paligemma_outputs = paligemma_model.vision_tower(pixel_values=pixel_values)
    feature = paligemma_outputs.last_hidden_state
    hf_feature = hf_paligemma_outputs.last_hidden_state

    torch.testing.assert_close(hf_feature, feature, rtol=1e-5, atol=1e-8)

    proj_features = paligemma_model.multi_modal_projector(feature)
    hf_proj_features = hf_paligemma_model.multi_modal_projector(hf_feature)

    torch.testing.assert_close(proj_features, hf_proj_features, rtol=1e-5, atol=1e-8)
