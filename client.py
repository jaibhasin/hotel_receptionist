# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hotel Receptionist Environment Client — GODMODE Edition.

This client connects to the hotel receptionist RL environment server
over WebSocket and provides a clean Python API for RL training loops.

=== How it fits in the architecture ===

    ┌──────────────┐   WebSocket   ┌──────────────────────────┐
    │  RL Training  │◄────────────►│  Environment Server      │
    │  Loop (uses   │   JSON msgs  │  (FastAPI + WebSocket)   │
    │  this client) │              │                          │
    └──────────────┘              └──────────────────────────┘

    The client handles:
      - WebSocket connection management (connect/disconnect)
      - Serializing actions to JSON for the server
      - Deserializing server responses into typed Observation objects
      - Docker container lifecycle (optional auto-start)

=== Usage Example ===

    from hotel_receptionist.client import HotelReceptionistEnv
    from hotel_receptionist.models import HotelReceptionistAction

    # Connect to a running server
    with HotelReceptionistEnv(base_url="http://localhost:8000") as client:

        # Start a new episode (random scenario + guest)
        result = client.reset()
        print(f"Scenario: {result.observation.scenario_type}")
        print(f"Guest says: {result.observation.guest_message}")

        # Take an action (greet the guest)
        action = HotelReceptionistAction(
            action_type="greet",
            message="Welcome to the Grand Hotel! How may I help you today?",
        )
        result = client.step(action)
        print(f"Reward: {result.observation.reward_breakdown}")
        print(f"Guest responds: {result.observation.guest_message}")
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# Try relative import first (when loaded as part of the hotel_receptionist package by the server)
# Fall back to absolute import (when inference.py imports client.py directly in the validator)
try:
    from .models import HotelReceptionistAction, HotelReceptionistObservation
except ImportError:
    from models import HotelReceptionistAction, HotelReceptionistObservation


class HotelReceptionistEnv(
    EnvClient[HotelReceptionistAction, HotelReceptionistObservation, State]
):
    """
    WebSocket client for the Hotel Receptionist RL Environment.

    Maintains a persistent connection to the environment server,
    enabling low-latency multi-step interactions for training.

    Each client instance gets its own environment session on the server,
    so multiple agents can train in parallel with separate clients.

    Methods:
      - reset() → StepResult: start a new episode
      - step(action) → StepResult: send an action, get observation + reward
      - close(): disconnect from the server
      - from_docker_image(image): auto-start a container and connect
    """

    def _step_payload(self, action: HotelReceptionistAction) -> Dict:
        """
        Convert the action object into a JSON-serializable dict for the server.

        The server expects a flat dict with the action_type, message,
        and any optional parameters. None-valued fields are excluded
        to keep the payload clean.

        Args:
            action: HotelReceptionistAction to serialize

        Returns:
            Dict ready for JSON encoding and WebSocket transmission
        """
        # Start with required fields
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
            "message": action.message,
        }

        # Add optional fields only if they have values
        # (keeps the WebSocket payload compact)
        # New fields: urgency_level, loyalty_points_awarded, upgrade_room_type
        # add richer semantics for emergency escalations, loyalty comp, and upgrades.
        optional_fields = [
            "room_number", "discount_percent", "compensation_details",
            "reservation_details", "service_details", "lost_item_description",
            "department", "internal_notes",
            "urgency_level", "loyalty_points_awarded", "upgrade_room_type",
        ]
        for field in optional_fields:
            value = getattr(action, field, None)
            if value is not None:
                payload[field] = value

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[HotelReceptionistObservation]:
        """
        Parse the server's JSON response into a typed StepResult.

        The server sends back a nested dict with observation data,
        reward, and done flag. This method reconstructs the full
        HotelReceptionistObservation with all its rich fields.

        Args:
            payload: raw JSON dict from the server

        Returns:
            StepResult containing the observation, reward, and done flag
        """
        obs_data = payload.get("observation", {})

        # Build the full observation from server data
        observation = HotelReceptionistObservation(
            # Scenario context
            scenario_type=obs_data.get("scenario_type", ""),
            scenario_difficulty=obs_data.get("scenario_difficulty", 1),
            # Guest info
            guest_message=obs_data.get("guest_message", ""),
            guest_profile=obs_data.get("guest_profile", {}),
            # Hotel state
            hotel_state=obs_data.get("hotel_state", {}),
            # Conversation
            conversation_history=obs_data.get("conversation_history", []),
            # Actions
            available_actions=obs_data.get("available_actions", []),
            # Rewards
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            # System
            system_notifications=obs_data.get("system_notifications", []),
            time_of_day=obs_data.get("time_of_day", "morning"),
            current_date=obs_data.get("current_date", ""),
            # Turns
            turn_number=obs_data.get("turn_number", 0),
            max_turns=obs_data.get("max_turns", 10),
            # Hints
            hints=obs_data.get("hints", []),
            # Standard RL fields
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse the server's state response into a State object.

        Used for querying the environment's internal state (episode ID,
        step count) without taking an action.

        Args:
            payload: JSON response from the /state endpoint

        Returns:
            State with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
