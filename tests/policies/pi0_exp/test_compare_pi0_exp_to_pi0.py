"""
uv run pytest -s tests/policies/pi0_exp/test_compare_pi0_exp_to_pi0.py

This file ensures that the default configuration of pi0_exp matches pi0.
"""

import random
from dataclasses import fields

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0_exp.configuration_pi0_exp import PI0ConfigExp
from tests.utils import DEVICE

BATCH_SIZE = 2
CHUNK_SIZE = 50


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_noise(shape, device):
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )
    return noise


def sample_time(bsize, device):
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32, device=device)


def sample_beta(alpha, beta, bsize, device):
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([beta]))
    return m.sample((bsize,)).to(device).reshape((bsize,))


@pytest.fixture(scope="module")
def dummy_dataset_metadata(lerobot_dataset_metadata_factory, info_factory, tmp_path_factory):
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
    tmp_path = tmp_path_factory.mktemp("pi0_test")
    ds_meta = lerobot_dataset_metadata_factory(root=tmp_path / "init", info=info)
    return ds_meta


@pytest.fixture(scope="module")
def pi0(dummy_dataset_metadata):
    set_seed(42)
    policy_cfg = PI0Config(
        pretrained_path="brandonyang/pi0_droid",
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
    )

    features = dataset_to_policy_features(dummy_dataset_metadata.features)
    policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    policy_cfg.input_features = {
        key: ft for key, ft in features.items() if key not in policy_cfg.output_features
    }

    policy = make_policy(policy_cfg, ds_meta=dummy_dataset_metadata)
    return policy.to(DEVICE)


@pytest.fixture(scope="module")
def pi0_exp(dummy_dataset_metadata):
    set_seed(42)
    policy_cfg = PI0ConfigExp(
        pretrained_path="brandonyang/pi0_droid",
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
    )

    features = dataset_to_policy_features(dummy_dataset_metadata.features)
    policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    policy_cfg.input_features = {
        key: ft for key, ft in features.items() if key not in policy_cfg.output_features
    }

    policy = make_policy(policy_cfg, ds_meta=dummy_dataset_metadata)
    return policy.to(DEVICE)


def test_config_equal(pi0, pi0_exp):
    assert isinstance(pi0.config, PI0Config), "pi0.config is not of type PI0Config"
    assert isinstance(pi0_exp.config, PI0ConfigExp), "pi0_exp.config is not of type PI0ConfigExp"
    assert pi0_exp.config.type == "pi0_exp", "pi0_exp.config.type is not 'pi0_exp'"
    for field in fields(pi0.config):
        name = field.name
        assert hasattr(pi0_exp.config, name), f"Missing field in pi0_exp.config: {name}"

        val1 = getattr(pi0.config, name)
        val2 = getattr(pi0_exp.config, name)

        assert val1 == val2, f"Field '{name}' mismatch: {val1} != {val2}"


def test_weights_equal(pi0, pi0_exp):
    state_dict1 = pi0.state_dict()
    state_dict2 = pi0_exp.state_dict()

    for key, value1 in state_dict1.items():
        assert key in state_dict2, f"Key '{key}' not found in pi0_exp state_dict."
        value2 = state_dict2[key]
        torch.testing.assert_close(value1, value2, rtol=1e-5, atol=1e-5)


def test_select_action(pi0, pi0_exp):
    set_seed(42)
    batch = {
        "observation.state": torch.randn(BATCH_SIZE, 8, device=DEVICE),
        "observation.images.external_cam": torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE),
        "observation.images.wrist_cam": torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE),
        "task": ["Do something." for _ in range(BATCH_SIZE)],
    }
    noise = sample_noise(
        (BATCH_SIZE, CHUNK_SIZE, pi0.config.max_action_dim), device=DEVICE
    )  # noise is modified in place in the diffusion loop
    noise_copy = noise.clone()

    pi0_actions = pi0.select_action(batch, noise=noise)
    pi0_exp_actions = pi0_exp.select_action(batch, noise=noise_copy)

    torch.testing.assert_close(pi0_actions, pi0_exp_actions, rtol=1e-5, atol=1e-5)


def test_forward_function(pi0, pi0_exp):
    set_seed(42)
    batch = {
        "observation.state": torch.randn(BATCH_SIZE, 8, device=DEVICE),
        "action": torch.randn(BATCH_SIZE, CHUNK_SIZE, 8, device=DEVICE),
        "observation.images.external_cam": torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE),
        "observation.images.wrist_cam": torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE),
        "task": ["Do something." for _ in range(BATCH_SIZE)],
    }
    noise = sample_noise((BATCH_SIZE, CHUNK_SIZE, pi0.config.max_action_dim), device=DEVICE)
    time = sample_time(BATCH_SIZE, device=DEVICE)

    pi0_res = pi0.forward(batch, noise=noise, time=time)
    pi0_exp_res = pi0_exp.forward(batch, noise=noise, time=time)

    torch.testing.assert_close(pi0_res, pi0_exp_res, rtol=1e-5, atol=1e-5)
