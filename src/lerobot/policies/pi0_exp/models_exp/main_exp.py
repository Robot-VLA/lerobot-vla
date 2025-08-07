"""
Adapted from Huggingface v0.50.3 transformers library.


"""

from dataclasses import dataclass, field

import torch.nn as nn

from lerobot.policies.pi0_exp.models_exp.gemma_exp import GemmaConfig, GemmaModel
from lerobot.policies.pi0_exp.models_exp.siglip_exp import SiglipVisionConfig, SiglipVisionModel


@dataclass
class PaliGemmaConfig:
    text_config: GemmaConfig = field(
        default_factory=lambda: GemmaConfig(
            _attn_implementation_autoset=True,
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=2,
            eos_token_id=1,
            head_dim=256,
            hidden_act="gelu_pytorch_tanh",
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=2048,
            initializer_range=0.02,
            intermediate_size=16384,
            max_position_embeddings=8192,
            model_type="gemma",
            num_attention_heads=8,
            num_hidden_layers=18,
            num_image_tokens=256,
            num_key_value_heads=1,
            pad_token_id=0,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            torch_dtype="float32",
            transformers_version="4.50.3",
            use_cache=True,
            vocab_size=257152,
        )
    )
    vision_config: SiglipVisionConfig = field(default_factory=SiglipVisionConfig)


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True
        )

    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config

        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config=config.text_config)


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config

        self.model = GemmaModel(config)


if __name__ == "__main__":
    paligemma_cfg = PaliGemmaConfig()
    expert_cfg = GemmaConfig()

    paligemma = PaliGemmaForConditionalGeneration(paligemma_cfg)
    gemma_expert = GemmaForCausalLM(expert_cfg)
