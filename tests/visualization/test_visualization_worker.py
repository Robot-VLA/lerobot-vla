"""
uv run pytest -s tests/visualization/test_visualization_worker.py
"""

import time

import torch

from lerobot.visualization.worker import VisualizationWorker
from tests.utils import DEVICE


def test_visualization_worker():
    images = [
        torch.rand(1, 3, 224, 224, dtype=torch.float32, device=DEVICE) * 2 - 1,
        torch.rand(1, 3, 224, 224, dtype=torch.float32, device=DEVICE) * 2 - 1,
    ]
    steps = [torch.rand(1, 30, 32, dtype=torch.float32, device=DEVICE) for _ in range(10)]

    visualizer = VisualizationWorker(endpoint_url="ws://localhost:8765")
    visualizer.enqueue(images, steps)

    time.sleep(1)
