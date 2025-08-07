"""
uv run python tests/visualization/ws_server.py

This file simply prints any message received over a WebSocket connection.
This file is used with `tests/visualization/test_visualization_worker.py`
"""

import asyncio

import websockets


async def echo_server(websocket):
    message = await websocket.recv()
    print("[Server received]:", message)


async def main():
    async with websockets.serve(echo_server, "localhost", 8765):
        print("WebSocket server running on ws://localhost:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
