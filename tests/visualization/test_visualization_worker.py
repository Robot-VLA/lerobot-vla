"""
uv run pytest -s tests/visualization/test_visualization_worker.py
"""

import asyncio
import json
import threading
import time

import pytest
import torch
import websockets

from lerobot.visualization.worker import VisualizationWorker
from tests.utils import DEVICE

RECEIVED_MESSAGES = []


@pytest.fixture(scope="function")
def websocket_test_server():
    """Start a local WebSocket echo server and collect received messages."""
    RECEIVED_MESSAGES.clear()

    async def echo_server(websocket):
        try:
            message = await websocket.recv()
            RECEIVED_MESSAGES.append(message)
        except websockets.ConnectionClosed:
            pass

    async def run_server():
        async with websockets.serve(echo_server, "localhost", 8765):
            await asyncio.Event().wait()  # Keep server running forever

    thread = threading.Thread(target=asyncio.run, args=(run_server(),), daemon=True)
    thread.start()

    asyncio.run(asyncio.sleep(0.1))  # Give server time to start

    yield RECEIVED_MESSAGES  # Provide access to received messages in test

    # Server will shut down automatically at end of test since it's daemonized


def test_visualization_worker(websocket_test_server):
    images = [
        torch.rand(1, 3, 224, 224, dtype=torch.float32, device=DEVICE) * 2 - 1,
        torch.rand(1, 3, 224, 224, dtype=torch.float32, device=DEVICE) * 2 - 1,
    ]
    steps = [torch.rand(1, 30, 32, dtype=torch.float32, device=DEVICE) for _ in range(10)]

    worker = VisualizationWorker(endpoint_url="ws://localhost:8765")
    worker.enqueue(images, steps)

    time.sleep(1)  # Give background thread time to send

    assert websocket_test_server, "No message received by WebSocket server."
    data = json.loads(websocket_test_server[0])
    assert "images" in data
    assert "steps" in data
