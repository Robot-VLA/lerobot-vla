"""
uv run pytest -s tests/envs/test_action_space_conversion.py
"""

import torch

from lerobot.envs.isaaclab.isaaclab_env import repack_delta_to_absolute


def test_repack_delta_to_absolute_batch():
    actions = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], [[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]]]
    )
    state = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 100]], [[10, 20, 30, 40, 50, 60, 70, 200]]])

    # Mask 1: update first 7 dimensions
    mask1 = torch.tensor([True, True, True, True, True, True, True, False])
    expected1 = torch.tensor(
        [[[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 0.8]], [[9.9, 19.8, 29.7, 39.6, 49.5, 59.4, 69.3, -0.8]]]
    )
    out1 = repack_delta_to_absolute(actions, state, mask1)
    torch.testing.assert_close(out1, expected1, rtol=1e-4, atol=1e-6)

    # Mask 2: update only dims 0 and 2
    mask2 = torch.tensor([True, False, True, False, False, False, False, False])
    expected2 = torch.tensor(
        [[[1.1, 0.2, 3.3, 0.4, 0.5, 0.6, 0.7, 0.8]], [[9.9, -0.2, 29.7, -0.4, -0.5, -0.6, -0.7, -0.8]]]
    )
    out2 = repack_delta_to_absolute(actions, state, mask2)
    torch.testing.assert_close(out2, expected2, rtol=1e-4, atol=1e-6)
