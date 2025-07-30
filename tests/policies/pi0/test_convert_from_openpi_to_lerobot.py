"""
uv run pytest -s tests/policies/pi0/test_convert_from_openpi_to_lerobot.py

Expected result:

E           Failed: 10/1600 values differ by > 3.0e-02 (max diff=0.0329)
E            index      lerobot      openpi      diff
E           -------------------------------------------
E           [0,25,9]        0.1127      0.1432      0.0305
E           [0,26,9]        0.1022      0.1331      0.0309
E           [0,28,9]        0.0830      0.1153      0.0323
E           [0,29,9]        0.0849      0.1171      0.0322
E           [0,30,9]        0.0835      0.1147      0.0312
E           [0,31,9]        0.0848      0.1160      0.0312
E           [0,32,9]        0.0862      0.1173      0.0311
E           [0,46,8]       -0.1684     -0.1378      0.0307
E           [0,47,8]       -0.1760     -0.1453      0.0307
E           [0,49,8]       -0.1910     -0.1581      0.0329
"""

import einops
import jax
import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType
from lerobot.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.conversion_scripts.convert_pi0_to_hf_lerobot import (
    load_gemma_expert_weights,
    load_paligemma_weights,
    load_projector_weights,
)
from lerobot.policies.pi0.conversion_scripts.openpi.src.openpi.models import model as _model
from lerobot.policies.pi0.conversion_scripts.openpi.src.openpi.models import pi0
from lerobot.policies.pi0.conversion_scripts.openpi.src.openpi.shared import download
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from tests.utils import DEVICE

ATOL = 3e-2  # Absolute tolerance for action differences
ATTN_IMPL = "sdpa"  # Attention implementation to use in the policy


def modify_policy(orig_policy, openpi_state_dict):
    print("Modifying policy with OpenPI state dict...")

    orig_policy.to(DEVICE)

    load_paligemma_weights(orig_policy, openpi_state_dict)
    load_gemma_expert_weights(orig_policy, openpi_state_dict)
    load_projector_weights(orig_policy, openpi_state_dict)


@pytest.fixture
def dummy_dataset_metadata(lerobot_dataset_metadata_factory, info_factory, tmp_path):
    camera_features = {
        "observation.images.base_0_rgb": {
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
            "info": None,
        },
        "observation.images.left_wrist_0_rgb": {
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
            "info": None,
        },
        "observation.images.right_wrist_0_rgb": {
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
            "info": None,
        },
    }
    # These will be padded
    motor_features = {
        "action": {
            "dtype": "float32",
            "shape": (32,),
            "names": [f"motor_{i}" for i in range(32)],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (32,),
            "names": [f"state_{i}" for i in range(32)],
        },
    }
    info = info_factory(
        total_episodes=1, total_frames=1, camera_features=camera_features, motor_features=motor_features
    )
    ds_meta = lerobot_dataset_metadata_factory(root=tmp_path / "init", info=info)
    return ds_meta


@pytest.fixture
def openpi_state_dict():
    return _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))


@pytest.fixture
def lerobot_pi0(dummy_dataset_metadata, monkeypatch, openpi_state_dict):
    policy_cfg = PI0Config(
        n_action_steps=50,
        chunk_size=50,
        attention_implementation=ATTN_IMPL,
    )
    features = dataset_to_policy_features(dummy_dataset_metadata.features)
    policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    policy_cfg.input_features = {
        key: ft for key, ft in features.items() if key not in policy_cfg.output_features
    }

    policy = PI0Policy(policy_cfg)
    modify_policy(policy, openpi_state_dict)

    def mock_prepare_images(self, batch):
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # HACK: We skip normalization here as we only use dummy values
            # img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def mock_prepare_language(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_STATE].device
        b_size = batch[OBS_STATE].shape[0]

        # NOTE: Match BaseModelConfig.fake_obs() to generate 1's for language tokens and masks
        lang_tokens = torch.ones((b_size, self.config.tokenizer_max_length), dtype=torch.int64, device=device)
        lang_masks = torch.ones((b_size, self.config.tokenizer_max_length), dtype=torch.bool, device=device)

        return lang_tokens, lang_masks

    @torch.no_grad()
    def mock_select_action(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mock select_action method to return all actions from chunk instead of sampling."""
        self.eval()

        # HACK: We skip normalization here as we only use dummy values
        # batch = self.normalize_inputs(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        # HACK: We skip normalization here as we only use dummy values
        # actions = self.unnormalize_outputs({"action": actions})["action"]

        return actions

    monkeypatch.setattr(type(policy), "prepare_language", mock_prepare_language)
    monkeypatch.setattr(type(policy), "select_action", mock_select_action)
    monkeypatch.setattr(type(policy), "prepare_images", mock_prepare_images)

    return policy


@pytest.fixture
def openpi_pi0(openpi_state_dict):
    config = pi0.Pi0Config()
    model = config.load(openpi_state_dict)
    return model


@pytest.fixture
def get_policy_inputs():
    config = pi0.Pi0Config()
    batch_size = 1
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    # Create lerobot inputs from openpi fake_obs and fake_act
    def _preprocess_image(img: torch.Tensor) -> torch.Tensor:
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)
        # img = img / 255.0  # NOTE: We do not normalize images here, as we use fake values
        return img

    fake_obs_dict = obs.to_dict()
    batch = {}
    batch[OBS_STATE] = fake_obs_dict["state"]
    batch[f"{OBS_IMAGES}.base_0_rgb"] = fake_obs_dict["image"]["base_0_rgb"]
    batch[f"{OBS_IMAGES}.left_wrist_0_rgb"] = fake_obs_dict["image"]["left_wrist_0_rgb"]
    batch[f"{OBS_IMAGES}.right_wrist_0_rgb"] = fake_obs_dict["image"]["right_wrist_0_rgb"]
    batch[ACTION] = act

    for k in batch:
        if k.startswith(OBS_IMAGES):
            batch[k] = _preprocess_image(torch.from_numpy(np.array(batch[k])).to(DEVICE))
        else:
            batch[k] = torch.from_numpy(np.array(batch[k])).to(DEVICE)

    return {
        "openpi_inputs": {
            "obs": obs,
            "act": act,
        },
        "lerobot_inputs": batch,
    }


def test_openpi_pi0(openpi_pi0, lerobot_pi0, get_policy_inputs):
    key = jax.random.key(0)
    batch_size = get_policy_inputs["lerobot_inputs"][OBS_STATE].shape[0]

    noise = jax.random.normal(
        key, (batch_size, lerobot_pi0.config.n_action_steps, lerobot_pi0.config.max_action_dim)
    )
    torch_noise = torch.from_numpy(np.array(noise)).to(DEVICE)

    lerobot_actions = lerobot_pi0.select_action(get_policy_inputs["lerobot_inputs"], noise=torch_noise)
    pi0_actions = openpi_pi0.sample_actions(key, get_policy_inputs["openpi_inputs"]["obs"])
    pi0_torch = torch.from_numpy(np.array(pi0_actions)).to(lerobot_actions.device)

    # --- custom assertion with full table output ---
    atol = ATOL
    diff = torch.abs(lerobot_actions - pi0_torch)
    mask = diff > atol

    num_diff = int(mask.sum())
    total = diff.numel()
    if num_diff > 0:
        # collect all mismatch entries
        bad_idxs = mask.nonzero(as_tuple=False)
        rows = []
        for idx in bad_idxs:
            i, j, k = idx.tolist()
            v_lr = lerobot_actions[i, j, k].item()
            v_p0 = pi0_torch[i, j, k].item()
            rows.append((f"[{i},{j},{k}]", v_lr, v_p0, abs(v_lr - v_p0)))

        # build a table string
        percentage = num_diff / total * 100
        header = (
            f"{num_diff}/{total} ({percentage}) values differ by > {atol:.1e} (max diff={float(diff[mask].max()):.4f})\n"
        )
        header += " index      lerobot      openpi      diff\n"
        header += "-------------------------------------------\n"
        for idx_str, v_lr, v_p0, d in rows:
            header += f"{idx_str:10s}  {v_lr:10.4f}  {v_p0:10.4f}  {d:10.4f}\n"

        pytest.fail(header)
    # else test passes
