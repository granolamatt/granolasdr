#!/usr/bin/env python3
"""
ZMQ → WebSocket bridge for the granolasdr dashboard.

Subscribes to the ZMQ XPUB proxy (default port 5600), forwards every message
to all connected browser WebSocket clients.

Message framing:
  ZMQ: two frames — [topic_bytes, payload_bytes]
  WebSocket to browser:
    • text topics (ft8/decode, js8/decode, ft8/timing, js8/timing):
        JSON string with "topic" field merged in, e.g.
        {"topic":"ft8/decode","call":"W1AW","freq":14074000,...}
    • "waterfall" topic: raw binary frame (2048 bytes, browser detects ArrayBuffer)

Usage:
    python3 ws_bridge.py [--xpub PORT] [--ws-port PORT] [--host HOST]
"""

import argparse
import asyncio
import json
import zmq
import zmq.asyncio

BINARY_TOPICS = {b"waterfall"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpub", type=int, default=5600, help="ZMQ XPUB port (default 5600)")
    ap.add_argument("--ws-port", type=int, default=8765, help="WebSocket port (default 8765)")
    ap.add_argument("--host", default="0.0.0.0", help="WebSocket bind host (default 0.0.0.0)")
    args = ap.parse_args()

    try:
        import websockets
    except ImportError:
        raise SystemExit("pip install websockets")

    ctx = zmq.asyncio.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{args.xpub}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")   # all topics

    clients: set = set()

    async def handle_client(ws):
        clients.add(ws)
        print(f"[ws_bridge] client connected  ({len(clients)} total)")
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)
            print(f"[ws_bridge] client disconnected  ({len(clients)} total)")

    async def forward():
        while True:
            frames = await sub.recv_multipart()
            if len(frames) != 2:
                continue
            topic, payload = frames

            if topic in BINARY_TOPICS:
                msg = payload          # send as binary WebSocket frame
            else:
                try:
                    data = json.loads(payload)
                    data["topic"] = topic.decode()
                    msg = json.dumps(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

            dead = set()
            for ws in list(clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            clients.difference_update(dead)

    server = await websockets.serve(handle_client, args.host, args.ws_port)
    print(f"[ws_bridge] ZMQ XPUB  tcp://localhost:{args.xpub}")
    print(f"[ws_bridge] WebSocket  ws://{args.host}:{args.ws_port}")

    await asyncio.gather(server.serve_forever(), forward())


if __name__ == "__main__":
    asyncio.run(main())
