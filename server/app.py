# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Hotel Receptionist RL Environment — GODMODE Edition.

This module creates the HTTP + WebSocket server that exposes the environment
to RL training clients. It's the bridge between the simulation engine and
the outside world.

=== Server Architecture ===

    ┌─────────────────────────────────────────────────────┐
    │                  FastAPI Server                      │
    │                                                     │
    │  HTTP Endpoints (for simple interactions):          │
    │    POST /reset   → start new episode                │
    │    POST /step    → send action, get observation      │
    │    GET  /state   → current environment state         │
    │    GET  /schema  → action/observation JSON schemas   │
    │                                                     │
    │  WebSocket Endpoint (for persistent RL sessions):   │
    │    WS /ws        → persistent bidirectional channel  │
    │                    (lower latency, session state)    │
    │                                                     │
    │  Supports up to 10 concurrent training sessions     │
    │  (each gets an independent environment instance)    │
    └─────────────────────────────────────────────────────┘

=== Running the Server ===

    # Development (auto-reload on code changes):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production (multiple worker processes):
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Direct execution:
    python -m server.app
    # or:
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import HotelReceptionistAction, HotelReceptionistObservation
    from .hotel_receptionist_environment import HotelReceptionistEnvironment
except ImportError:
    from models import HotelReceptionistAction, HotelReceptionistObservation
    from server.hotel_receptionist_environment import HotelReceptionistEnvironment


# Create the FastAPI app with the OpenEnv server framework
# max_concurrent_envs=10 allows 10 parallel training sessions via WebSocket
# Each WebSocket client gets its own HotelReceptionistEnvironment instance
app = create_app(
    HotelReceptionistEnvironment,       # the environment class (instantiated per session)
    HotelReceptionistAction,            # the action schema (validated on /step)
    HotelReceptionistObservation,       # the observation schema (returned from /step)
    env_name="hotel_receptionist",      # display name in the API docs
    max_concurrent_envs=10,             # GODMODE: support 10 parallel training sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for running the server directly (without uvicorn CLI).

    Usage:
        uv run --project . server
        uv run --project . server --port 8001
        python -m hotel_receptionist.server.app

    Args:
        host: network interface to bind to ("0.0.0.0" = all interfaces)
        port: TCP port to listen on (default 8000)
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # When run directly, parse CLI args and start the server
    # Supports: python -m server.app --port 8001
    main()
