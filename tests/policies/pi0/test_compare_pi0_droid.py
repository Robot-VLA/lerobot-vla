"""
uv run pytest -s tests/policies/pi0/test_compare_pi0_droid.py

- This file uses mocked make_droid_example() and compare generation results for lerobot's pytorch implementation
and the original openpi implementation.
- This file also tests the end-to-end normalization and unnormalization logic.


Expected result:

3/80 (3.75) values differ by > 3.0e-02 (max diff=0.0468)
 index      lerobot      openpi      diff
-------------------------------------------
[0,7,6]         0.1041      0.0672      0.0369
[0,8,6]         0.1740      0.1297      0.0443
[0,9,6]         0.2025      0.1557      0.0468
"""

import einops
import jax
import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType
from lerobot.constants import OBS_IMAGES, OBS_STATE
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.conversion_scripts.openpi.src.openpi.policies import droid_policy
from lerobot.policies.pi0.conversion_scripts.openpi.src.openpi.policies import policy_config as _policy_config
from lerobot.policies.pi0.conversion_scripts.openpi.src.openpi.shared import download
from lerobot.policies.pi0.conversion_scripts.openpi.src.openpi.training import config as _config
from tests.utils import DEVICE

ATOL = 3e-2  # Absolute tolerance for action differences
ATTN_IMPL = "sdpa"  # Attention implementation to use in the policy
THRESHOLD_PERCENT = 5.0


@pytest.fixture
def dummy_dataset_metadata(lerobot_dataset_metadata_factory, info_factory, tmp_path):
    camera_features = {
        "observation.images.external_cam": {
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
            "info": None,
        },
        "observation.images.wrist_cam": {
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
            "info": None,
        },
    }
    # These will be padded
    motor_features = {
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"motor_{i}" for i in range(8)],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"state_{i}" for i in range(8)],
        },
    }
    info = info_factory(
        total_episodes=1, total_frames=1, camera_features=camera_features, motor_features=motor_features
    )
    ds_meta = lerobot_dataset_metadata_factory(root=tmp_path / "init", info=info)
    return ds_meta


@pytest.fixture
def openpi_pi0():
    config = _config.get_config("pi0_droid")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_droid")
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    return policy


@pytest.fixture
def lerobot_pi0(monkeypatch, dummy_dataset_metadata):
    policy_cfg = PI0Config(
        pretrained_path="brandonyang/pi0_droid",
        n_action_steps=10,  # Default values for "pi0_droid" in openpi
        attention_implementation="sdpa",
    )
    features = dataset_to_policy_features(dummy_dataset_metadata.features)
    policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    policy_cfg.input_features = {
        key: ft for key, ft in features.items() if key not in policy_cfg.output_features
    }

    policy = make_policy(policy_cfg, ds_meta=dummy_dataset_metadata)

    @torch.no_grad()
    def mock_select_action(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return all n_action_steps actions"""
        self.eval()

        print("LeRobot before normalizing state: ", batch[OBS_STATE])
        batch = self.normalize_inputs(batch)
        print("LeRobot after normalizing state: ", batch[OBS_STATE])

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({"action": actions})["action"]

        return actions

    monkeypatch.setattr(type(policy), "select_action", mock_select_action)

    return policy


@pytest.fixture
def get_policy_inputs():
    openpi_example = droid_policy.make_droid_example(seed=42)
    lerobot_batch = {}

    def _preprocess_image(img: torch.Tensor) -> torch.Tensor:
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)
        if img.max() > 1.0:
            img /= 255
        return img

    lerobot_batch[OBS_STATE] = np.concatenate(
        [openpi_example["observation/joint_position"], openpi_example["observation/gripper_position"]]
    )
    lerobot_batch[f"{OBS_IMAGES}.external_cam"] = openpi_example["observation/exterior_image_1_left"]
    lerobot_batch[f"{OBS_IMAGES}.wrist_cam"] = openpi_example["observation/wrist_image_left"]
    lerobot_batch["task"] = [openpi_example["prompt"]]

    for k in lerobot_batch:
        if k.startswith(OBS_IMAGES):
            lerobot_batch[k] = _preprocess_image(torch.from_numpy(lerobot_batch[k]).unsqueeze(0).to(DEVICE))
        elif k.startswith(OBS_STATE):
            lerobot_batch[k] = torch.from_numpy(lerobot_batch[k]).unsqueeze(0).to(DEVICE).to(torch.float32)

    return {"lerobot_inputs": lerobot_batch, "openpi_inputs": openpi_example}


def test_openpi_pi0(openpi_pi0, lerobot_pi0, get_policy_inputs):
    key = jax.random.key(0)
    batch_size = get_policy_inputs["lerobot_inputs"][OBS_STATE].shape[0]
    noise = jax.random.normal(
        key, (batch_size, lerobot_pi0.config.n_action_steps, lerobot_pi0.config.max_action_dim)
    )
    torch_noise = torch.from_numpy(np.array(noise)).to(DEVICE)

    lerobot_actions = lerobot_pi0.select_action(get_policy_inputs["lerobot_inputs"], noise=torch_noise)
    openpi_actions = openpi_pi0.infer(get_policy_inputs["openpi_inputs"], rng=key)["actions"]
    pi0_torch = torch.from_numpy(openpi_actions).unsqueeze(0).to(lerobot_actions.device)
    print("Open pi actions shape: ", pi0_torch.shape)
    print("Lerobot shape: ", lerobot_actions.shape)

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
        header = f"{num_diff}/{total} ({percentage}) values differ by > {atol:.1e} (max diff={float(diff[mask].max()):.4f})\n"
        header += " index      lerobot      openpi      diff\n"
        header += "-------------------------------------------\n"
        for idx_str, v_lr, v_p0, d in rows:
            header += f"{idx_str:10s}  {v_lr:10.4f}  {v_p0:10.4f}  {d:10.4f}\n"

        print(f"\033[93m{header}\033[0m")

        if percentage > THRESHOLD_PERCENT:
            pytest.fail(f"Action mismatch exceeds {THRESHOLD_PERCENT:.1f}% threshold")
