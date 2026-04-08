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

=== REQUIRED ENV VARS ===
    API_BASE_URL   The API endpoint for the LLM (default: HF Router)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

=== STDOUT FORMAT (exact, required by judges) ===
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

=== HOW TO RUN ===
    export HF_TOKEN="hf_your_token_here"
    export API_BASE_URL="https://router.huggingface.co/v1"   # optional
    export MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"   # optional
    python inference.py
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

from models import HotelReceptionistAction, HotelReceptionistObservation
from server.hotel_receptionist_environment import HotelReceptionistEnvironment


# ──────────────────────────────────────────────────────────────
#  Configuration — all three must come from env vars per the
#  hackathon checklist. Defaults are provided so the script
#  still runs in development without them set.
# ──────────────────────────────────────────────────────────────

# LLM API endpoint (OpenAI-compatible). Judges override via API_BASE_URL.
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"

# Model to use for inference. Judges override via MODEL_NAME.
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-70B-Instruct"

# HF token — mandatory, no default (script will exit if missing)
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Environment / benchmark label used in [START] log lines
BENCHMARK = "hotel_receptionist"

# Max steps per episode — keeps runtime under the 20-minute limit
MAX_STEPS = 10

# Score threshold: a task is "successful" if the average reward exceeds this
SUCCESS_SCORE_THRESHOLD = 0.4


# ──────────────────────────────────────────────────────────────
#  Named Tasks — easy → medium → hard
#
#  A fixed seed makes each task deterministic and reproducible
#  across runs, which is required by the evaluation criteria.
#
#  The "grader" for each task is the environment's built-in reward
#  function (rule-based Accuracy + Efficiency, LLM-judged
#  Professionalism + Empathy), which always returns [0.0, 1.0].
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
    """
    Emit the [START] line at the beginning of each task episode.

    Format: [START] task=<name> env=<benchmark> model=<model_name>
    """
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """
    Emit one [STEP] line immediately after env.step() returns.

    Format: [STEP] step=<n> action=<str> reward=<0.00> done=<bool> error=<str|null>
    Rules:
      - reward formatted to 2 decimal places
      - done is lowercase true/false
      - error is the error message string, or null if no error
    """
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Escape the action string so it fits on one line (no internal newlines)
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """
    Emit the [END] line after each task episode completes (always, even on error).

    Format: [END] success=<bool> steps=<n> score=<0.000> rewards=<r1,r2,...>
    Rules:
      - success is lowercase true/false
      - score formatted to 3 decimal places
      - rewards is a comma-separated list, each to 2 decimal places
    """
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

    Packs all relevant context (scenario type, guest profile, conversation
    history, available actions, hotel state) into a single readable block.

    Args:
        obs: latest observation returned by env.reset() or env.step()

    Returns:
        Formatted prompt string the LLM uses to choose its next action
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

    Args:
        client: OpenAI client pointed at the HF Router
        obs:    current environment observation

    Returns:
        HotelReceptionistAction ready to pass to env.step()
    """
    user_prompt = build_user_prompt(obs)
    try:
        # Call the LLM — uses OpenAI SDK but points at HF Router
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
        # Safe fallback: polite generic reply so the episode keeps running
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
    env: HotelReceptionistEnvironment,
    client: OpenAI,
    task: dict,
) -> dict:
    """
    Run one named task (a complete episode) and emit the required logs.

    Flow:
      1. [START] log emitted
      2. env.reset(seed=fixed_seed) → deterministic scenario
      3. Loop MAX_STEPS times:
           LLM picks action → env.step(action) → [STEP] log
      4. env.close() in finally block
      5. [END] log emitted (always, even if an exception occurred)

    Score = mean(step_rewards), clamped to [0.0, 1.0].
    Divides by actual steps taken so early resolution isn't penalized.
    Each step reward is already in [0.0, 1.0] from the environment's
    reward function, so the final score is always in [0.0, 1.0].

    Args:
        env:    HotelReceptionistEnvironment instance (in-process)
        client: OpenAI client for LLM calls
        task:   dict with task_id, seed, description

    Returns:
        Dict with task_id, score, success, steps, rewards
    """
    task_id = task["task_id"]
    seed = task["seed"]

    # Emit [START] before anything else
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # reset() with a fixed seed → same scenario every run (deterministic grader)
        obs = env.reset(seed=seed)

        print(f"[DEBUG] {task_id}: {obs.scenario_type} diff={obs.scenario_difficulty}/5 "
              f"guest={obs.guest_profile.get('name')} mood={obs.guest_profile.get('mood')}",
              flush=True, file=sys.stderr)

        for step in range(1, MAX_STEPS + 1):
            # Episode finished early (environment set done=True)
            if obs.done:
                break

            # LLM agent decides what to do
            action = get_agent_action(client, obs)

            # Step the environment: apply action → get observation + reward
            try:
                obs = env.step(action)
                reward = float(obs.reward or 0.0)
                done = bool(obs.done)
                error = None
            except Exception as step_exc:
                # Record the error but don't crash — emit [STEP] then [END]
                reward = 0.0
                done = True
                error = str(step_exc)[:80]

            rewards.append(reward)
            steps_taken = step

            # Emit [STEP] immediately after env.step() — required format
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
        # Mean reward across actual steps taken, clamped to [0.0, 1.0].
        # Dividing by actual steps (not MAX_STEPS) so an agent that resolves
        # a task in 3 steps with reward 0.9 scores 0.9, not 0.27.
        actual_steps = len(rewards)
        score = sum(rewards) / actual_steps if actual_steps > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # Always close the environment, even if an exception occurred
        try:
            env.close()
        except Exception as close_exc:
            print(f"[DEBUG] env.close() error: {close_exc}", flush=True, file=sys.stderr)

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
#  Main — runs all 3 tasks and prints an aggregate summary
# ──────────────────────────────────────────────────────────────

async def main() -> None:
    """
    Entry point: validate env vars, run all 3 tasks, print summary.

    Uses asyncio.run() to match the sample inference script pattern,
    even though the environment itself is synchronous. This keeps the
    script compatible with async client variants (from_docker_image etc.)
    """
    # ── Validate required env vars ───────────────────────────
    if not API_KEY:
        print("ERROR: HF_TOKEN environment variable is required.", flush=True)
        print("  export HF_TOKEN='hf_your_token_here'", flush=True)
        sys.exit(1)

    # ── Initialize LLM client ────────────────────────────────
    # OpenAI SDK pointed at the HF Router — no OpenAI key needed.
    # API_BASE_URL and MODEL_NAME come from env vars (set at module level).
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 60, flush=True)
    print("  HOTEL RECEPTIONIST — INFERENCE EVALUATION", flush=True)
    print("=" * 60, flush=True)
    print(f"  API:    {API_BASE_URL}", flush=True)
    print(f"  Model:  {MODEL_NAME}", flush=True)
    print(f"  Tasks:  {len(TASKS)} (easy / medium / hard)", flush=True)
    print("=" * 60, flush=True)

    # ── Run each task sequentially ───────────────────────────
    # Each task gets its own environment instance for clean state isolation.
    # A fresh env per task also means env.close() is safe to call after each.
    all_results = []
    for task in TASKS:
        print(f"\n--- Running task: {task['task_id']} ---", flush=True)
        print(f"    {task['description']}", flush=True)

        # New environment instance per task for clean state
        env = HotelReceptionistEnvironment()
        result = await run_task(env, client, task)
        all_results.append(result)

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
