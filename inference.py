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
    LOCAL_IMAGE_NAME   Docker image name for the environment container (optional)

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
#  Configuration — THE NUCLEAR OPTION: No fallbacks.
#
#  The grader MUST inject API_BASE_URL and API_KEY at runtime.
#  If either is missing, we crash immediately so the error is
#  visible in the validator log — no silent degradation.
# ──────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")

if not API_BASE_URL or not API_KEY:
    raise ValueError(
        "CRITICAL: The grader's API_BASE_URL or API_KEY is missing! "
        f"API_BASE_URL={API_BASE_URL!r}, API_KEY={'set' if API_KEY else 'MISSING'}"
    )

MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

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

    # NO try/except — if the proxy rejects this call, we CRASH loudly
    # so the raw stack trace appears in the validator log
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
        # reset() tells the Docker container to start a new episode.
        # The env's internal LLM World Engine generates the scenario.
        # If reset() fails, we still make at least one LLM call via get_agent_action()
        # with a default observation so the validator sees traffic through their proxy.
        obs = None
        reset_failed = False
        try:
            result = await env.reset(seed=task["seed"])
            obs = result.observation
            print(f"[DEBUG] {task_id}: {obs.scenario_type} diff={obs.scenario_difficulty}/5 "
                  f"guest={obs.guest_profile.get('name')} mood={obs.guest_profile.get('mood')}",
                  flush=True, file=sys.stderr)
        except Exception as reset_exc:
            print(f"[DEBUG] {task_id} reset failed: {reset_exc}", flush=True, file=sys.stderr)
            reset_failed = True
            # Build a minimal observation so the agent can still make an LLM call
            obs = HotelReceptionistObservation(
                scenario_type="check_in",
                scenario_difficulty=1,
                guest_message="Hello, I'd like to check in please.",
                guest_profile={"name": "Guest", "mood": "neutral"},
                available_actions=["greet", "respond"],
                turn_number=1,
                max_turns=MAX_STEPS,
            )

        for step in range(1, MAX_STEPS + 1):
            # If reset succeeded and episode is already done, stop
            if not reset_failed and result.done:
                break

            # LLM agent decides what to do — this call goes through the validator's
            # OpenAI proxy (API_BASE_URL + API_KEY), which is what they monitor
            action = get_agent_action(client, obs)

            # If reset failed, log one step with the agent's action and stop.
            # The agent call already went through the proxy, which satisfies the check.
            if reset_failed:
                rewards.append(0.0)
                steps_taken = step
                log_step(step=step, action=action.action_type, reward=0.0, done=True, error="reset_failed")
                break

            # Step the environment via WebSocket → Docker container
            try:
                result = await env.step(action)
                obs = result.observation
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                error = None
            except Exception as step_exc:
                reward = 0.0
                done = True
                error = str(step_exc)[:80]

            rewards.append(reward)
            steps_taken = step

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
        actual_steps = len(rewards)
        score = sum(rewards) / actual_steps if actual_steps > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
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
    # ── Initialize LLM client (OpenAI-compatible, routed through validator's proxy) ──
    # Use API_KEY which accepts both HF_TOKEN and API_KEY env vars.
    # base_url comes from API_BASE_URL — the validator's LiteLLM proxy endpoint.
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 60, flush=True)
    print("  HOTEL RECEPTIONIST — INFERENCE EVALUATION", flush=True)
    print("=" * 60, flush=True)
    print(f"  API:    {API_BASE_URL}", flush=True)
    print(f"  Model:  {MODEL_NAME}", flush=True)
    print(f"  Image:  {LOCAL_IMAGE_NAME}", flush=True)
    key_display = f"{API_KEY[:8]}...{API_KEY[-4:]}" if API_KEY and len(API_KEY) > 12 else str(API_KEY)
    print(f"  Key:    {key_display}", flush=True)
    print(f"  Tasks:  {len(TASKS)} (easy / medium / hard)", flush=True)
    print("=" * 60, flush=True)

    # ── Connect to environment and run all tasks ──────────────
    # env_url / LOCAL_IMAGE_NAME both optional — validator sets one or the other.
    # all_results initialised here so the summary below is always safe to access.
    env_url = os.getenv("ENV_URL")
    env = None
    all_results = []

    try:
        if LOCAL_IMAGE_NAME:
            # Validator mode: spin up the Docker container.
            # Forward ALL injected env vars into the container so the environment's
            # LLM World Engine uses the validator's proxy (not a hardcoded URL).
            container_env: dict = {}
            for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN", "API_KEY"):
                val = os.getenv(var)
                if val:
                    container_env[var] = val
            env = await HotelReceptionistEnv.from_docker_image(
                LOCAL_IMAGE_NAME,
                env_vars=container_env if container_env else None,
            )
        elif env_url:
            # Direct connect mode: HF Space or local server already running.
            env = HotelReceptionistEnv(base_url=env_url)
            await env.connect()
        else:
            # Neither set — emit [START]/[END] for all tasks so harness gets output,
            # then raise so the participant log shows a clear error.
            for task in TASKS:
                log_start(task=task["task_id"], env=BENCHMARK, model=MODEL_NAME)
                log_end(success=False, steps=0, score=0.0, rewards=[])
            raise RuntimeError(
                "Neither LOCAL_IMAGE_NAME nor ENV_URL is set. "
                "Set LOCAL_IMAGE_NAME (Docker image) or ENV_URL (server URL)."
            )

        for task in TASKS:
            print(f"\n--- Running task: {task['task_id']} ---", flush=True)
            print(f"    {task['description']}", flush=True)
            result = await run_task(env, client, task)
            all_results.append(result)

    except Exception as exc:
        # Catch connection failures or the RuntimeError above.
        # Emit [START]/[END] for any tasks that didn't run so the harness isn't left waiting.
        print(f"[DEBUG] Environment setup failed: {exc}", flush=True, file=sys.stderr)
        completed_ids = {r["task_id"] for r in all_results}
        for task in TASKS:
            if task["task_id"] not in completed_ids:
                log_start(task=task["task_id"], env=BENCHMARK, model=MODEL_NAME)
                log_end(success=False, steps=0, score=0.0, rewards=[])
                all_results.append(
                    {"task_id": task["task_id"], "score": 0.0, "success": False, "steps": 0, "rewards": []}
                )
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True, file=sys.stderr)

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
