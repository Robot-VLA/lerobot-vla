import asyncio
import base64
import io
import json
import queue
import threading
import time

import numpy as np
import torch
import websockets
from PIL import Image


class VisualizationWorker:
    """A background worker that sends image data to a WebSocket server."""

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, images: list[torch.Tensor], steps: list[torch.Tensor] | torch.Tensor):
        """Non-blocking enqueue of raw data."""
        self._queue.put((images, steps))

    def _run(self):
        asyncio.run(self._websocket_loop())

    async def _websocket_loop(self):
        while True:
            images, steps = self._queue.get()
            payload = self._prepare_payload(images, steps)
            try:
                async with websockets.connect(self.endpoint_url) as ws:
                    await ws.send(json.dumps(payload))
            except Exception as e:
                print(f"[VisWorker] WS send failed: {e}")

    def _prepare_payload(self, images: list[torch.Tensor], steps: list[torch.Tensor] | torch.Tensor):
        """Convert image tensors and steps to JSON-friendly format."""
        encoded_images = []
        for img in images:
            img = img.detach().cpu()

            # Remove batch dimension if shape is [1, C, H, W]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)

            if img.ndim == 3 and img.shape[0] in (1, 3):  # (C, H, W)
                img = img.permute(1, 2, 0)  # (H, W, C)

            img_np = img.numpy()
            if img_np.min() < 0:
                img_np = (img_np + 1.0) / 2.0

            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            buf = io.BytesIO()
            Image.fromarray(img_np).save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            encoded_images.append(encoded)

        if isinstance(steps, list):
            steps = torch.stack(steps)
        steps_list = steps.detach().cpu().tolist()

        return {"images": encoded_images, "steps": steps_list, "timestamp": time.time()}
