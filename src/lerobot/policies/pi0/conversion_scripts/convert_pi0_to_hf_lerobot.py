import numpy as np
import torch

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.conversion_scripts.openpi.src.openpi.models import model as _model
from lerobot.policies.pi0.conversion_scripts.openpi.src.openpi.shared import download
from lerobot.policies.pi0.modeling_pi0 import PI0Policy


def to_torch(t):
    val = torch.from_numpy(np.array(t))
    return val.to(torch.float32)


def load_paligemma_weights(orig_policy: PI0Policy, openpi_state_dict: dict):
    orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.patch_embedding.load_state_dict(
        {
            "weight": to_torch(openpi_state_dict["PaliGemma"]["img"]["embedding"]["kernel"]).permute(
                3, 2, 0, 1
            ),
            "bias": to_torch(openpi_state_dict["PaliGemma"]["img"]["embedding"]["bias"]),
        },
        strict=True,
    )
    orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.position_embedding.load_state_dict(
        {"weight": to_torch(openpi_state_dict["PaliGemma"]["img"]["pos_embedding"]).squeeze(0)},
        strict=True,
    )

    encoderblock_layernorm0_scale = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"]["LayerNorm_0"]["scale"]
    )
    encoderblock_layernorm0_bias = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"]["LayerNorm_0"]["bias"]
    )
    encoderblock_layernorm1_scale = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"]["LayerNorm_1"]["scale"]
    )
    encoderblock_layernorm1_bias = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"]["LayerNorm_1"]["bias"]
    )

    encoderblock_mlp_dense0_kernel = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"]["MlpBlock_0"]["Dense_0"][
            "kernel"
        ]
    )
    encoderblock_mlp_dense0_bias = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"]["MlpBlock_0"]["Dense_0"]["bias"]
    )
    encoderblock_mlp_dense1_kernel = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"]["MlpBlock_0"]["Dense_1"][
            "kernel"
        ]
    )
    encoderblock_mlp_dense1_bias = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"]["MlpBlock_0"]["Dense_1"]["bias"]
    )

    encoderblock_attention_0_key_kernel = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"][
            "MultiHeadDotProductAttention_0"
        ]["key"]["kernel"]
    )
    encoderblock_attention_0_key_bias = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"][
            "MultiHeadDotProductAttention_0"
        ]["key"]["bias"]
    )
    encoderblock_attention_0_value_kernel = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"][
            "MultiHeadDotProductAttention_0"
        ]["value"]["kernel"]
    )
    encoderblock_attention_0_value_bias = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"][
            "MultiHeadDotProductAttention_0"
        ]["value"]["bias"]
    )
    encoderblock_attention_0_query_kernel = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"][
            "MultiHeadDotProductAttention_0"
        ]["query"]["kernel"]
    )
    encoderblock_attention_0_query_bias = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"][
            "MultiHeadDotProductAttention_0"
        ]["query"]["bias"]
    )
    encoderblock_attention_0_out_kernel = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"][
            "MultiHeadDotProductAttention_0"
        ]["out"]["kernel"]
    )
    encoderblock_attention_0_out_bias = to_torch(
        openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoderblock"][
            "MultiHeadDotProductAttention_0"
        ]["out"]["bias"]
    )

    for i in range(
        orig_policy.model.paligemma_with_expert.config.paligemma_config.vision_config.num_hidden_layers
    ):
        orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers[
            i
        ].layer_norm1.load_state_dict(
            {"weight": encoderblock_layernorm0_scale[i], "bias": encoderblock_layernorm0_bias[i]},
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers[
            i
        ].layer_norm2.load_state_dict(
            {"weight": encoderblock_layernorm1_scale[i], "bias": encoderblock_layernorm1_bias[i]},
            strict=True,
        )

        orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers[
            i
        ].mlp.fc1.load_state_dict(
            {"weight": encoderblock_mlp_dense0_kernel[i].t(), "bias": encoderblock_mlp_dense0_bias[i]},
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers[
            i
        ].mlp.fc2.load_state_dict(
            {"weight": encoderblock_mlp_dense1_kernel[i].t(), "bias": encoderblock_mlp_dense1_bias[i]},
            strict=True,
        )

        # TODO: check if the transpose is correct
        in_features, num_heads, head_dim = encoderblock_attention_0_key_kernel[i].shape
        orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers[
            i
        ].self_attn.k_proj.load_state_dict(
            {
                "weight": encoderblock_attention_0_key_kernel[i]
                .reshape(in_features, num_heads * head_dim)
                .t(),
                "bias": encoderblock_attention_0_key_bias[i].reshape(num_heads * head_dim),
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers[
            i
        ].self_attn.v_proj.load_state_dict(
            {
                "weight": encoderblock_attention_0_value_kernel[i]
                .reshape(in_features, num_heads * head_dim)
                .t(),
                "bias": encoderblock_attention_0_value_bias[i].reshape(num_heads * head_dim),
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers[
            i
        ].self_attn.q_proj.load_state_dict(
            {
                "weight": encoderblock_attention_0_query_kernel[i]
                .reshape(in_features, num_heads * head_dim)
                .t(),
                "bias": encoderblock_attention_0_query_bias[i].reshape(num_heads * head_dim),
            },
            strict=True,
        )
        # TODO: check if the transpose is correct
        orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers[
            i
        ].self_attn.out_proj.load_state_dict(
            {
                "weight": encoderblock_attention_0_out_kernel[i]
                .reshape(num_heads * head_dim, in_features)
                .t(),
                "bias": encoderblock_attention_0_out_bias[i],
            },
            strict=True,
        )

    orig_policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model.post_layernorm.load_state_dict(
        {
            "weight": to_torch(openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoder_norm"]["scale"]),
            "bias": to_torch(openpi_state_dict["PaliGemma"]["img"]["Transformer"]["encoder_norm"]["bias"]),
        },
        strict=True,
    )
    # multi_modal_projector
    orig_policy.model.paligemma_with_expert.paligemma.multi_modal_projector.linear.load_state_dict(
        {
            "weight": to_torch(openpi_state_dict["PaliGemma"]["img"]["head"]["kernel"]).t(),
            "bias": to_torch(openpi_state_dict["PaliGemma"]["img"]["head"]["bias"]),
        },
        strict=True,
    )
    # text decoder
    orig_policy.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.load_state_dict(
        {"weight": to_torch(openpi_state_dict["PaliGemma"]["llm"]["embedder"]["input_embedding"])},
        strict=True,
    )

    # pop the einsum attention + mlp representations. There are 18 layers in gemma-2b.

    # (18, 8, 256, 2048)
    llm_attention_attn_vec_einsum = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["attn"]["attn_vec_einsum"]["w"]
    )
    # (18, 2, 1, 2048, 256)
    llm_attention_kv_einsum = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["attn"]["kv_einsum"]["w"]
    )
    # (18, 8, 2048, 256)
    llm_attention_q_einsum = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["attn"]["q_einsum"]["w"]
    )
    # (18, 2, 2048, 16384)
    llm_mlp_gating_einsum = to_torch(openpi_state_dict["PaliGemma"]["llm"]["layers"]["mlp"]["gating_einsum"])
    # (18, 16384, 2048)
    llm_mlp_linear = to_torch(openpi_state_dict["PaliGemma"]["llm"]["layers"]["mlp"]["linear"])
    # (18, 2048)
    llm_input_layernorm = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["pre_attention_norm"]["scale"]
    )
    # (18, 2048)
    llm_post_attention_layernorm = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["pre_ffw_norm"]["scale"]
    )

    config = orig_policy.model.paligemma_with_expert.config.paligemma_config
    for i in range(
        orig_policy.model.paligemma_with_expert.config.paligemma_config.text_config.num_hidden_layers
    ):
        orig_policy.model.paligemma_with_expert.paligemma.language_model.model.layers[
            i
        ].self_attn.q_proj.load_state_dict(
            {
                "weight": llm_attention_q_einsum[i]
                .permute(0, 2, 1)
                .reshape(
                    config.text_config.num_attention_heads * config.text_config.head_dim,
                    config.text_config.hidden_size,
                )
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.language_model.model.layers[
            i
        ].self_attn.k_proj.load_state_dict(
            {"weight": llm_attention_kv_einsum[i, 0, 0].t()},
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.language_model.model.layers[
            i
        ].self_attn.v_proj.load_state_dict(
            {"weight": llm_attention_kv_einsum[i, 1, 0].t()},
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.language_model.model.layers[
            i
        ].self_attn.o_proj.load_state_dict(
            {
                "weight": llm_attention_attn_vec_einsum[i]
                .permute(2, 0, 1)
                .reshape(
                    config.text_config.hidden_size,
                    config.text_config.num_attention_heads * config.text_config.head_dim,
                )
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.language_model.model.layers[
            i
        ].mlp.gate_proj.load_state_dict(
            {
                "weight": llm_mlp_gating_einsum[i, 0].t(),
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.language_model.model.layers[
            i
        ].mlp.up_proj.load_state_dict(
            {
                "weight": llm_mlp_gating_einsum[i, 1].t(),
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.language_model.model.layers[
            i
        ].mlp.down_proj.load_state_dict(
            {
                "weight": llm_mlp_linear[i].t(),
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.language_model.model.layers[
            i
        ].input_layernorm.load_state_dict(
            {
                "weight": llm_input_layernorm[i],
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.paligemma.language_model.model.layers[
            i
        ].post_attention_layernorm.load_state_dict(
            {
                "weight": llm_post_attention_layernorm[i],
            },
            strict=True,
        )

    orig_policy.model.paligemma_with_expert.paligemma.language_model.model.norm.load_state_dict(
        {"weight": to_torch(openpi_state_dict["PaliGemma"]["llm"]["final_norm"]["scale"])},
        strict=True,
    )


def load_gemma_expert_weights(orig_policy: PI0Policy, openpi_state_dict: dict):
    # torch.Size([18, 8, 256, 1024])
    llm_attention_attn_vec_einsum_1 = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["attn"]["attn_vec_einsum_1"]["w"]
    )
    # torch.Size([18, 2, 1, 1024, 256])
    llm_attention_kv_einsum_1 = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["attn"]["kv_einsum_1"]["w"]
    )
    # torch.Size([18, 8, 1024, 256])
    llm_attention_q_einsum_1 = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["attn"]["q_einsum_1"]["w"]
    )
    # torch.Size([18, 2, 1024, 4096])
    llm_mlp_gating_einsum_1 = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["mlp_1"]["gating_einsum"]
    )
    # torch.Size([18, 4096, 1024])
    llm_mlp_linear_1 = to_torch(openpi_state_dict["PaliGemma"]["llm"]["layers"]["mlp_1"]["linear"])
    # torch.Size([18, 1024])
    llm_input_layernorm_1 = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["pre_attention_norm_1"]["scale"]
    )
    # torch.Size([18, 1024])
    llm_post_attention_layernorm_1 = to_torch(
        openpi_state_dict["PaliGemma"]["llm"]["layers"]["pre_ffw_norm_1"]["scale"]
    )

    config = orig_policy.model.paligemma_with_expert.config.gemma_expert_config
    for i in range(config.num_hidden_layers):
        orig_policy.model.paligemma_with_expert.gemma_expert.model.layers[i].self_attn.q_proj.load_state_dict(
            {
                "weight": llm_attention_q_einsum_1[i]
                .permute(0, 2, 1)
                .reshape(
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                )
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.gemma_expert.model.layers[i].self_attn.k_proj.load_state_dict(
            {"weight": llm_attention_kv_einsum_1[i, 0, 0].t()},
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.gemma_expert.model.layers[i].self_attn.v_proj.load_state_dict(
            {"weight": llm_attention_kv_einsum_1[i, 1, 0].t()},
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.gemma_expert.model.layers[i].self_attn.o_proj.load_state_dict(
            {
                "weight": llm_attention_attn_vec_einsum_1[i]
                .permute(2, 0, 1)
                .reshape(
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                )
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.gemma_expert.model.layers[i].mlp.gate_proj.load_state_dict(
            {
                "weight": llm_mlp_gating_einsum_1[i, 0].t(),
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.gemma_expert.model.layers[i].mlp.up_proj.load_state_dict(
            {
                "weight": llm_mlp_gating_einsum_1[i, 1].t(),
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.gemma_expert.model.layers[i].mlp.down_proj.load_state_dict(
            {
                "weight": llm_mlp_linear_1[i].t(),
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.gemma_expert.model.layers[i].input_layernorm.load_state_dict(
            {
                "weight": llm_input_layernorm_1[i],
            },
            strict=True,
        )
        orig_policy.model.paligemma_with_expert.gemma_expert.model.layers[
            i
        ].post_attention_layernorm.load_state_dict(
            {
                "weight": llm_post_attention_layernorm_1[i],
            },
            strict=True,
        )
    orig_policy.model.paligemma_with_expert.gemma_expert.model.norm.load_state_dict(
        {"weight": to_torch(openpi_state_dict["PaliGemma"]["llm"]["final_norm_1"]["scale"])},
        strict=True,
    )


def load_projector_weights(orig_policy: PI0Policy, openpi_state_dict: dict):
    orig_policy.model.state_proj.load_state_dict(
        {
            "weight": to_torch(openpi_state_dict["state_proj"]["kernel"]).t(),
            "bias": to_torch(openpi_state_dict["state_proj"]["bias"]),
        },
        strict=True,
    )
    orig_policy.model.action_in_proj.load_state_dict(
        {
            "weight": to_torch(openpi_state_dict["action_in_proj"]["kernel"]).t(),
            "bias": to_torch(openpi_state_dict["action_in_proj"]["bias"]),
        },
        strict=True,
    )
    orig_policy.model.action_out_proj.load_state_dict(
        {
            "weight": to_torch(openpi_state_dict["action_out_proj"]["kernel"]).t(),
            "bias": to_torch(openpi_state_dict["action_out_proj"]["bias"]),
        },
        strict=True,
    )
    orig_policy.model.action_time_mlp_in.load_state_dict(
        {
            "weight": to_torch(openpi_state_dict["action_time_mlp_in"]["kernel"]).t(),
            "bias": to_torch(openpi_state_dict["action_time_mlp_in"]["bias"]),
        },
        strict=True,
    )
    orig_policy.model.action_time_mlp_out.load_state_dict(
        {
            "weight": to_torch(openpi_state_dict["action_time_mlp_out"]["kernel"]).t(),
            "bias": to_torch(openpi_state_dict["action_time_mlp_out"]["bias"]),
        },
        strict=True,
    )


if __name__ == "__main__":
    policy_cfg = PI0Config()
    policy = PI0Policy(policy_cfg)

    openpi_state_dict = _model.restore_params(
        download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params")
    )

    # openpi weights are float32, so we store the policy in float32 on huggingface
    policy.to(torch.float32)

    load_paligemma_weights(policy, openpi_state_dict)
    load_gemma_expert_weights(policy, openpi_state_dict)
    load_projector_weights(policy, openpi_state_dict)

    policy.save_pretrained(
        save_directory="./pi0-checkpoint",
        repo_id="brandonyang/pi0",
        push_to_hub=True,
    )
