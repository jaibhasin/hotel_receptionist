#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hotel Receptionist RL Environment — Hackathon Inference Script.

Runs 3 named tasks (easy → medium → hard) against the hotel receptionist
environment and emits the required structured stdout logs for evaluation.

=== HOW IT WORKS ===
    1. The environment runs inside a Docker container (HF Space)
    2. This script connects to it via WebSocket using HotelReceptionistEnv client
    3. The LLM agent (OpenAI client → HF Router) decides actions
    4. The environment scores actions and generates guest replies internally

=== REQUIRED ENV VARS ===
    API_BASE_URL       The API endpoint for the LLM (default: HF Router)
    MODEL_NAME         The model identifier to use for inference
    HF_TOKEN           Your Hugging Face / API key
    IMAGE_NAME         Docker image name for the environment container

=== STDOUT FORMAT (exact, required by judges) ===
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ──────────────────────────────────────────────────────────────
#  Add project root to sys.path so local imports work when the
#  script is executed directly (e.g. python inference.py)
# ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the OpenEnv client (connects to env via WebSocket)
# NOT the raw environment class — the client talks to the Docker container
from client import HotelReceptionistEnv
from models import HotelReceptionistAction, HotelReceptionistObservation


# ──────────────────────────────────────────────────────────────
#  Configuration — env vars set by judges during validation
# ──────────────────────────────────────────────────────────────

# Docker image name for the environment container
IMAGE_NAME = os.getenv("IMAGE_NAME")

# LLM API endpoint (OpenAI-compatible). Judges override via API_BASE_URL.
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"

# Model to use for inference. Judges override via MODEL_NAME.
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-70B-Instruct"

# HF token — used for both LLM calls and env container auth
# Validator injects API_KEY pointing to their LiteLLM proxy — use it first.
# Fall back to HF_TOKEN for local testing only.
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

# Environment / benchmark label used in [START] log lines
BENCHMARK = "hotel_receptionist"

# Max steps per episode — keeps runtime under the 20-minute limit
MAX_STEPS = 10

# Score threshold: a task is "successful" if the average reward exceeds this
SUCCESS_SCORE_THRESHOLD = 0.4


# ──────────────────────────────────────────────────────────────
#  Named Tasks — easy → medium → hard
#
#  Each task runs a full episode. The environment generates the
#  scenario internally (via its LLM World Engine inside the
#  Docker container). The seed makes it deterministic.
# ──────────────────────────────────────────────────────────────

TASKS = [
    {
        # EASY: straightforward check-in, happy guest, difficulty 1
        "task_id": "easy_check_in",
        "seed": 42,
        "description": "Standard hotel check-in for a happy guest",
    },
    {
        # MEDIUM: billing dispute from an impatient guest, difficulty 3
        "task_id": "medium_complaint",
        "seed": 100,
        "description": "Resolve a billing dispute from an upset guest",
    },
    {
        # HARD: VIP arrival escalating into an emergency, difficulty 5
        "task_id": "hard_vip_emergency",
        "seed": 200,
        "description": "Handle a demanding VIP guest with an emergency situation",
    },
]


# ──────────────────────────────────────────────────────────────
#  Structured log helpers — produce EXACTLY the format required
#  by the hackathon evaluation harness.
#  Do NOT change field names, ordering, or formatting.
# ──────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line at the beginning of each task episode."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line immediately after env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Keep action on one line (no internal newlines)
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line after each task episode completes (always, even on error)."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────
#  LLM Agent — reads observations and chooses receptionist actions
# ──────────────────────────────────────────────────────────────

# System prompt that shapes the LLM's receptionist persona
SYSTEM_PROMPT = textwrap.dedent("""
    You are a professional hotel receptionist at the Grand Hotel.
    Handle guest interactions with professionalism, empathy, and efficiency.

    Respond in valid JSON with these fields:
    {
      "action_type": "<one of the available actions>",
      "message": "<your spoken response to the guest>",
      "room_number": "<optional: for assign_room / offer_upgrade>",
      "discount_percent": <optional: 0-50 for apply_discount>,
      "compensation_details": "<optional: for offer_compensation>",
      "reservation_details": {"check_in": "<YYYY-MM-DD>", "check_out": "<YYYY-MM-DD>", "room_type": "<standard|deluxe|suite|penthouse|accessible>", "guests": 1},
      "service_details": "<optional: for order_room_service / arrange_transport>",
      "lost_item_description": "<optional: for log_lost_item>",
      "department": "<optional: for transfer_call>",
      "internal_notes": "<optional: private hotel record>"
    }

    Guidelines:
    - First turn: always greet warmly with action_type="greet"
    - Angry/upset guests: apologize FIRST, then solve
    - VIP guests: acknowledge status, offer upgrades, use formal language
    - Emergencies: escalate immediately with call_security or escalate_manager
    - Use professional vocabulary: certainly, my pleasure, allow me, of course
    - Resolve within the turn limit; use end_interaction to close politely
""").strip()


def build_user_prompt(obs: HotelReceptionistObservation) -> str:
    """
    Convert the current environment observation into a text prompt for the LLM.

    Packs all relevant context (scenario, guest profile, conversation history,
    available actions, hotel state) into a single readable block.
    """
    # Guest profile summary
    profile = obs.guest_profile
    profile_summary = (
        f"Name: {profile.get('name', 'Unknown')}\n"
        f"Mood: {profile.get('mood', 'neutral')}\n"
        f"VIP: {'Yes' if profile.get('is_vip') else 'No'}"
        f"{' (' + profile.get('loyalty_tier', '') + ')' if profile.get('loyalty_tier') else ''}\n"
        f"Room preference: {profile.get('room_preference', 'any')}\n"
        f"Special requests: {', '.join(profile.get('special_requests', [])) or 'None'}\n"
        f"Nights: {profile.get('nights_staying', 1)} | Party: {profile.get('party_size', 1)}"
    )

    # Conversation history (last 6 turns to keep prompt concise)
    history_lines = [
        f"  {e.get('role', '?').capitalize()}: {e.get('message', '')}"
        for e in obs.conversation_history[-6:]
    ]
    history_str = "\n".join(history_lines) if history_lines else "  (conversation just started)"

    # Available room summary
    hotel = obs.hotel_state
    rooms_summary = ""
    for rtype, info in hotel.get("rooms_by_type", {}).items():
        count = info.get("available_count", 0)
        if count > 0:
            samples = info.get("sample_rooms", [])
            sample_str = ", ".join(
                f"#{r['room_number']} (fl {r['floor']}, ${r['price']:.0f}/night)"
                for r in samples[:2]
            )
            rooms_summary += f"  {rtype}: {count} available — {sample_str}\n"

    return textwrap.dedent(f"""
        SCENARIO: {obs.scenario_type} | Difficulty: {obs.scenario_difficulty}/5
        Time: {obs.time_of_day} | Date: {obs.current_date}
        Turn: {obs.turn_number}/{obs.max_turns}

        GUEST PROFILE:
        {profile_summary}

        GUEST SAYS: "{obs.guest_message}"

        CONVERSATION HISTORY:
        {history_str}

        AVAILABLE ACTIONS: {', '.join(obs.available_actions)}

        HOTEL AVAILABILITY ({hotel.get('available_rooms', '?')}/{hotel.get('total_rooms', '?')} rooms free):
        {rooms_summary.strip() or '  (no availability data)'}

        Respond with JSON containing "action_type" and "message" (plus optional fields).
    """).strip()


def get_agent_action(client: OpenAI, obs: HotelReceptionistObservation) -> HotelReceptionistAction:
    """
    Ask the LLM to choose a receptionist action given the current observation.

    Calls the OpenAI-compatible API at API_BASE_URL using MODEL_NAME.
    Falls back to a safe default action if the API call fails.
    """
    user_prompt = build_user_prompt(obs)
    try:
        # Call the LLM via HF Router (OpenAI-compatible endpoint)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=512,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True, file=sys.stderr)
        # Safe fallback so the episode keeps running
        raw = json.dumps({"action_type": "respond",
                          "message": "I apologize for the delay. Let me assist you right away."})

    # Parse the LLM's JSON response (handles markdown fences and extra text)
    text = raw
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {"action_type": "respond", "message": raw[:200]}

    # Ensure required fields are present
    data.setdefault("action_type", "respond")
    data.setdefault("message", "How may I help you?")

    return HotelReceptionistAction(
        action_type=data.get("action_type", "respond"),
        message=data.get("message", ""),
        room_number=data.get("room_number"),
        discount_percent=data.get("discount_percent"),
        compensation_details=data.get("compensation_details"),
        reservation_details=data.get("reservation_details"),
        service_details=data.get("service_details"),
        lost_item_description=data.get("lost_item_description"),
        department=data.get("department"),
        internal_notes=data.get("internal_notes"),
    )


# ──────────────────────────────────────────────────────────────
#  Task runner — runs one complete episode for a named task
# ──────────────────────────────────────────────────────────────

async def run_task(
    env: HotelReceptionistEnv,
    client: OpenAI,
    task: dict,
) -> dict:
    """
    Run one named task (a complete episode) and emit the required logs.

    Flow:
      1. [START] log
      2. env.reset() → get initial observation from Docker container
      3. Loop up to MAX_STEPS: LLM picks action → env.step(action) → [STEP] log
      4. env.close() in finally block
      5. [END] log (always emitted, even on exception)

    Score = mean(step_rewards), clamped to [0.0, 1.0].
    """
    task_id = task["task_id"]

    # Emit [START] before anything else
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # reset() tells the Docker container to start a new episode
        # The env's internal LLM World Engine generates the scenario
        # Pass seed for deterministic, reproducible scenarios across runs
        result = await env.reset(seed=task["seed"])
        obs = result.observation

        print(f"[DEBUG] {task_id}: {obs.scenario_type} diff={obs.scenario_difficulty}/5 "
              f"guest={obs.guest_profile.get('name')} mood={obs.guest_profile.get('mood')}",
              flush=True, file=sys.stderr)

        for step in range(1, MAX_STEPS + 1):
            # Episode finished early (environment set done=True)
            if result.done:
                break

            # LLM agent decides what to do
            action = get_agent_action(client, obs)

            # Step the environment via WebSocket → Docker container
            try:
                result = await env.step(action)
                obs = result.observation
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                error = None
            except Exception as step_exc:
                # Record the error but don't crash
                reward = 0.0
                done = True
                error = str(step_exc)[:80]

            rewards.append(reward)
            steps_taken = step

            # Emit [STEP] immediately after env.step()
            log_step(
                step=step,
                action=action.action_type,
                reward=reward,
                done=done,
                error=error,
            )

            if done or error:
                break

        # ── Compute final score ──────────────────────────────
        # Mean reward across actual steps, clamped to [0.0, 1.0]
        actual_steps = len(rewards)
        score = sum(rewards) / actual_steps if actual_steps > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        # Catch any exception (including RuntimeError from reset())
        # so we still emit [END] and don't crash the whole script
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True, file=sys.stderr)

    finally:
        # Always emit [END] — even if the episode failed
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


# ──────────────────────────────────────────────────────────────
#  Main — connect to env container, run all 3 tasks, print summary
# ──────────────────────────────────────────────────────────────

async def main() -> None:
    """
    Entry point: connect to environment Docker container via OpenEnv client,
    run all 3 tasks sequentially, print aggregate summary.

    Uses from_docker_image() to connect to the environment running in Docker,
    matching the pattern from the official sample inference script.
    """
    # ── Validate required env vars ───────────────────────────
    if not API_KEY:
        print("ERROR: HF_TOKEN environment variable is required.", flush=True)
        print("  export HF_TOKEN='hf_your_token_here'", flush=True)
        sys.exit(1)

    # ── Initialize LLM client (for the agent's decisions) ────
    # OpenAI SDK pointed at the HF Router — this is for the AGENT's LLM calls.
    # The environment's internal LLM calls happen inside the Docker container.
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 60, flush=True)
    print("  HOTEL RECEPTIONIST — INFERENCE EVALUATION", flush=True)
    print("=" * 60, flush=True)
    print(f"  API:    {API_BASE_URL}", flush=True)
    print(f"  Model:  {MODEL_NAME}", flush=True)
    print(f"  Image:  {IMAGE_NAME}", flush=True)
    print(f"  Key:    {API_KEY[:8]}...{API_KEY[-4:]}" if API_KEY and len(API_KEY) > 12 else f"  Key:    {API_KEY}", flush=True)
    print(f"  Tasks:  {len(TASKS)} (easy / medium / hard)", flush=True)
    print("=" * 60, flush=True)

    # Sanity check: make a test LLM call to verify the proxy is reachable
    # This call is NOT wrapped in try/except so we see the real error
    print("[DEBUG] Testing LLM proxy connection...", flush=True)
    try:
        test_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
        )
        print(f"[DEBUG] LLM proxy OK: {test_completion.choices[0].message.content}", flush=True)
    except Exception as e:
        print(f"[DEBUG] LLM proxy FAILED: {e}", flush=True)

    # ── Connect to the environment ─────────────────────────────
    # If IMAGE_NAME is set → use Docker (validator mode)
    # If ENV_URL is set → connect directly to a running server (local testing)
    # ENV_URL example: https://jai3-hotel-receptionist.hf.space
    env_url = os.getenv("ENV_URL")

    if IMAGE_NAME:
        # Validator mode: spin up Docker container with the environment
        env = await HotelReceptionistEnv.from_docker_image(IMAGE_NAME)
    elif env_url:
        # Local testing mode: connect to already-running HF Space or local server
        env = HotelReceptionistEnv(base_url=env_url)
        await env.connect()
    else:
        print("ERROR: Set IMAGE_NAME (Docker) or ENV_URL (direct connect).", flush=True)
        print("  For local testing: export ENV_URL='https://jai3-hotel-receptionist.hf.space'", flush=True)
        sys.exit(1)

    # ── Run each task sequentially ───────────────────────────
    all_results = []
    try:
        for task in TASKS:
            print(f"\n--- Running task: {task['task_id']} ---", flush=True)
            print(f"    {task['description']}", flush=True)

            result = await run_task(env, client, task)
            all_results.append(result)
    finally:
        # Always close the environment connection (cleans up Docker container)
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True, file=sys.stderr)

    # ── Aggregate summary ────────────────────────────────────
    print(f"\n{'=' * 60}", flush=True)
    print("  EVALUATION SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    total = len(all_results)
    n_success = sum(1 for r in all_results if r["success"])
    avg_score = sum(r["score"] for r in all_results) / total if total else 0.0

    print(f"  Tasks:      {total}", flush=True)
    print(f"  Successful: {n_success}/{total}", flush=True)
    print(f"  Avg score:  {avg_score:.3f}", flush=True)
    print(flush=True)

    for r in all_results:
        status = "SUCCESS" if r["success"] else "FAIL   "
        print(f"  [{status}] {r['task_id']:25s} score={r['score']:.3f} steps={r['steps']}", flush=True)

    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
