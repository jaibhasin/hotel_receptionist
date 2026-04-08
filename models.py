# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Hotel Receptionist RL Environment — GODMODE Edition.

This module defines every data structure the environment uses:
  - Guest profiles with personalities and preferences
  - Hotel room inventory and booking records
  - Rich action space (what the LLM-receptionist can do)
  - Detailed observation space (what the LLM sees after each action)

Architecture overview:
  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
  │  LLM Agent  │────▶│  Action (input)  │────▶│  Environment.step() │
  │ (the model) │◀────│ Observation (out) │◀────│  (hotel simulator)  │
  └─────────────┘     └──────────────────┘     └─────────────────────┘

The LLM receives an Observation describing the current situation (who is
at the desk, what they said, hotel state) and responds with an Action
(what to say, what operation to perform). The environment scores the
action on professionalism, accuracy, helpfulness, and efficiency.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ──────────────────────────────────────────────────────────────
#  Enums — define the discrete categories the environment uses
# ──────────────────────────────────────────────────────────────

class GuestMood(str, Enum):
    """
    How the guest is feeling when they approach the front desk.
    The receptionist's responses should adapt to the guest's mood —
    e.g. an ANGRY guest needs empathy first, a CONFUSED guest needs clarity.
    """
    HAPPY = "happy"
    NEUTRAL = "neutral"
    IMPATIENT = "impatient"
    ANGRY = "angry"
    CONFUSED = "confused"
    VIP_DEMANDING = "vip_demanding"     # high-status guest with high expectations
    ELDERLY_PATIENT = "elderly_patient" # needs extra patience and clarity


class ScenarioType(str, Enum):
    """
    The type of interaction happening at the front desk.
    Each scenario has different success criteria and reward weights.

    Example flow:
      1. Environment picks a ScenarioType (e.g. CHECK_IN)
      2. Generates a matching Guest profile + situation details
      3. LLM-receptionist must handle it via Actions
      4. Environment scores based on scenario-specific rubric
    """
    CHECK_IN = "check_in"                       # guest arriving, needs room assignment
    CHECK_OUT = "check_out"                     # guest leaving, needs billing + farewell
    RESERVATION = "reservation"                 # phone/walk-in reservation request
    COMPLAINT = "complaint"                     # something went wrong, guest is upset
    ROOM_SERVICE_REQUEST = "room_service"       # guest wants food/amenities delivered
    CONCIERGE_REQUEST = "concierge"             # directions, restaurant recs, tickets
    LOST_AND_FOUND = "lost_and_found"           # guest lost something in the hotel
    BILLING_DISPUTE = "billing_dispute"         # guest disagrees with charges
    EMERGENCY = "emergency"                     # fire alarm, medical issue, security
    VIP_ARRIVAL = "vip_arrival"                 # high-profile guest needs special treatment
    GROUP_BOOKING = "group_booking"             # conference/wedding block of rooms
    NOISE_COMPLAINT = "noise_complaint"         # guest disturbed by another guest
    ACCESSIBILITY_REQUEST = "accessibility"     # disability accommodation needed
    LATE_CHECKOUT = "late_checkout"             # guest wants to stay past checkout time
    LOYALTY_PROGRAM = "loyalty_program"         # points, upgrades, member benefits


class RoomType(str, Enum):
    """
    Room categories in the hotel, from budget to ultra-luxury.
    Each has different price points and amenities — the receptionist
    must match guests to appropriate rooms based on their booking/budget.
    """
    STANDARD = "standard"
    DELUXE = "deluxe"
    SUITE = "suite"
    PENTHOUSE = "penthouse"
    ACCESSIBLE = "accessible"   # ADA-compliant room


class RoomStatus(str, Enum):
    """
    Current state of a hotel room.
    The receptionist needs to know availability to assign rooms correctly.
    """
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    CLEANING = "cleaning"           # housekeeping in progress
    MAINTENANCE = "maintenance"     # out of order
    RESERVED = "reserved"           # booked but guest hasn't arrived


class ReceptionistActionType(str, Enum):
    """
    Every operation the LLM-receptionist can perform.

    The agent picks ONE action type per step, along with a natural-language
    response. Some actions require additional parameters (e.g. ASSIGN_ROOM
    needs a room_number). The environment validates that the action makes
    sense in context — assigning a room during a noise complaint loses points.
    """
    GREET = "greet"                         # welcome the guest
    RESPOND = "respond"                     # general conversational reply
    ASSIGN_ROOM = "assign_room"             # give guest a specific room
    PROCESS_CHECKOUT = "process_checkout"   # finalize departure + billing
    MAKE_RESERVATION = "make_reservation"   # book a future stay
    ESCALATE_MANAGER = "escalate_manager"   # hand off to a manager
    OFFER_UPGRADE = "offer_upgrade"         # suggest a better room
    APPLY_DISCOUNT = "apply_discount"       # give a price reduction
    CALL_SECURITY = "call_security"         # for emergencies/disturbances
    CALL_MAINTENANCE = "call_maintenance"   # fix something in a room
    ARRANGE_TRANSPORT = "arrange_transport" # taxi, shuttle, limo
    ORDER_ROOM_SERVICE = "order_room_service"
    PROVIDE_INFORMATION = "provide_information"  # answer a question
    APOLOGIZE = "apologize"                 # formal apology for issues
    OFFER_COMPENSATION = "offer_compensation"   # free night, voucher, etc.
    LOG_LOST_ITEM = "log_lost_item"         # record a lost item report
    TRANSFER_CALL = "transfer_call"         # route to another department
    END_INTERACTION = "end_interaction"      # politely close the conversation


# ──────────────────────────────────────────────────────────────
#  Guest & Room models — the "world state" the environment tracks
# ──────────────────────────────────────────────────────────────

class GuestProfile(Dict):
    """
    A simulated hotel guest. Generated by the environment's scenario engine.

    Fields:
      - name: guest's full name
      - mood: how they're feeling (affects reward for empathy)
      - is_vip: VIP guests expect premium treatment
      - loyalty_tier: None, "silver", "gold", "platinum"
      - has_reservation: whether they booked ahead
      - reservation_id: booking reference number (if applicable)
      - room_preference: what room type they want
      - special_requests: dietary needs, extra pillows, etc.
      - complaint_details: what's wrong (for complaint scenarios)
      - budget_sensitivity: "low" (price-conscious) to "high" (doesn't care)
      - language: primary language (for multilingual scenarios)
      - nights_staying: how many nights
      - party_size: number of guests in the party
    """
    pass  # We use a plain dict at runtime; this docstring is for clarity


class HotelRoom(Dict):
    """
    A single room in the hotel inventory.

    Fields:
      - room_number: e.g. "301"
      - room_type: RoomType value
      - status: RoomStatus value
      - floor: which floor (int)
      - price_per_night: float
      - amenities: list of strings ("ocean_view", "balcony", "jacuzzi", etc.)
      - max_occupancy: how many guests it holds
      - is_accessible: ADA-compliant features
      - current_guest: name of occupant (or None)
    """
    pass  # Plain dict at runtime


# ──────────────────────────────────────────────────────────────
#  Action — what the LLM sends to the environment each step
# ──────────────────────────────────────────────────────────────

class HotelReceptionistAction(Action):
    """
    The LLM-receptionist's response each turn.

    Every action has two parts:
      1. action_type — the operational intent (GREET, ASSIGN_ROOM, etc.)
      2. message — the natural language response spoken to the guest

    Optional fields provide parameters for specific action types:
      - room_number: required for ASSIGN_ROOM, OFFER_UPGRADE
      - discount_percent: used with APPLY_DISCOUNT (0-100)
      - compensation_details: what you're offering with OFFER_COMPENSATION
      - reservation_details: booking info for MAKE_RESERVATION
      - service_details: what to order for ORDER_ROOM_SERVICE
      - lost_item_description: item details for LOG_LOST_ITEM
      - department: where to route for TRANSFER_CALL
      - internal_notes: private notes (not shown to guest, used for logging)

    Example — checking in a guest:
        HotelReceptionistAction(
            action_type="assign_room",
            message="Welcome, Mr. Smith! I've prepared room 504 for you — "
                    "a deluxe suite with an ocean view, just as you requested.",
            room_number="504",
        )
    """

    # --- Required fields (every action needs these) ---
    action_type: str = Field(
        ...,
        description="What operation to perform (from ReceptionistActionType enum)"
    )
    message: str = Field(
        ...,
        description="Natural language response spoken to the guest"
    )

    # --- Optional fields (used by specific action types) ---
    room_number: Optional[str] = Field(
        default=None,
        description="Room number for ASSIGN_ROOM or OFFER_UPGRADE actions"
    )
    discount_percent: Optional[float] = Field(
        default=None,
        description="Discount percentage (0-100) for APPLY_DISCOUNT action"
    )
    compensation_details: Optional[str] = Field(
        default=None,
        description="What compensation to offer (free night, voucher, etc.)"
    )
    reservation_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Booking details for MAKE_RESERVATION (dates, room type, etc.)"
    )
    service_details: Optional[str] = Field(
        default=None,
        description="What to order/arrange for ROOM_SERVICE or ARRANGE_TRANSPORT"
    )
    lost_item_description: Optional[str] = Field(
        default=None,
        description="Description of lost item for LOG_LOST_ITEM"
    )
    department: Optional[str] = Field(
        default=None,
        description="Target department for TRANSFER_CALL"
    )
    internal_notes: Optional[str] = Field(
        default=None,
        description="Private notes for hotel records (not shown to guest)"
    )


# ──────────────────────────────────────────────────────────────
#  Observation — what the LLM receives from the environment
# ──────────────────────────────────────────────────────────────

class HotelReceptionistObservation(Observation):
    """
    Everything the LLM-receptionist can see after each step.

    This is the agent's "window into the world." It contains:
      - scenario_type: what kind of interaction this is
      - guest_message: what the guest just said
      - guest_profile: who the guest is (name, mood, VIP status, etc.)
      - hotel_state: room availability, occupancy rates, etc.
      - conversation_history: full dialogue so far
      - available_actions: which action types make sense right now
      - reward_breakdown: detailed scoring of the last action
      - system_notifications: alerts from the hotel (e.g. "Room 201 ready")
      - time_of_day / current_date: affects scenarios (late checkout, etc.)
      - scenario_difficulty: 1-5 star difficulty rating
      - turn_number: how many turns into this conversation
      - max_turns: conversation budget (efficiency matters!)
      - hints: optional nudges for the agent (can be disabled)

    The agent reads this observation, decides what to do, and returns
    a HotelReceptionistAction. The cycle repeats until the interaction
    ends (guest leaves, emergency resolved, etc.).
    """

    # --- Scenario context ---
    scenario_type: str = Field(
        default="",
        description="Current scenario type (from ScenarioType enum)"
    )
    scenario_difficulty: int = Field(
        default=1,
        description="Difficulty rating 1-5 (1=simple check-in, 5=VIP emergency)"
    )

    # --- Guest information ---
    guest_message: str = Field(
        default="",
        description="What the guest just said to the receptionist"
    )
    guest_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Guest details: name, mood, VIP status, preferences, etc."
    )

    # --- Hotel state ---
    hotel_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current hotel status: available rooms, occupancy, alerts"
    )

    # --- Conversation tracking ---
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Full dialogue history [{role: 'guest'/'receptionist', message: '...'}]"
    )

    # --- Action guidance ---
    available_actions: List[str] = Field(
        default_factory=list,
        description="Which action types are valid in the current context"
    )

    # --- Reward feedback ---
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Detailed scoring: {professionalism, accuracy, empathy, efficiency}"
    )

    # --- System info ---
    system_notifications: List[str] = Field(
        default_factory=list,
        description="Hotel system alerts (room ready, VIP arriving, etc.)"
    )
    time_of_day: str = Field(
        default="morning",
        description="Current time period: morning, afternoon, evening, night"
    )
    current_date: str = Field(
        default="",
        description="Simulated date (YYYY-MM-DD)"
    )

    # --- Turn management ---
    turn_number: int = Field(
        default=0,
        description="Current turn in this conversation"
    )
    max_turns: int = Field(
        default=10,
        description="Maximum turns allowed — resolve it before time runs out!"
    )

    # --- Learning aids ---
    hints: List[str] = Field(
        default_factory=list,
        description="Optional hints for the agent (e.g. 'Guest is VIP — offer upgrade')"
    )
