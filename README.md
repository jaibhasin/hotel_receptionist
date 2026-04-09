---
title: Hotel Receptionist Environment Server
emoji: 🏨
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
short_description: LLM-driven hotel simulation for RL agent training
tags:
  - openenv
---

# Hotel Receptionist Environment

A 100% LLM-driven hotel front-desk simulation where RL agents learn to handle realistic guest interactions — check-ins, billing disputes, VIP emergencies, and more — scored across professionalism, accuracy, empathy, and efficiency.

Agents receive rich observations (guest mood, hotel inventory, conversation history) and must choose from 18 receptionist actions. The environment generates unique scenarios via LLM and judges each agent response with a multi-dimensional reward.

## Quick Start

Run the baseline inference evaluation (3 tasks: easy → medium → hard):

```bash
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"   # optional
export API_BASE_URL="https://router.huggingface.co/v1"  # optional
python inference.py
```

Or use the environment directly in Python:

```python
from server.hotel_receptionist_environment import HotelReceptionistEnvironment
from models import HotelReceptionistAction

env = HotelReceptionistEnvironment()

# Reset to a new scenario (seed makes it reproducible)
obs = env.reset(seed=42)
print(f"Scenario: {obs.scenario_type} | Difficulty: {obs.scenario_difficulty}/5")
print(f"Guest says: {obs.guest_message}")

# Take an action
action = HotelReceptionistAction(
    action_type="greet",
    message="Welcome to the Grand Hotel! How may I assist you today?"
)
obs = env.step(action)
print(f"Reward: {obs.reward:.3f} | Done: {obs.done}")
print(f"Guest replies: {obs.guest_message}")

env.close()
```

## Environment Details

### Action Space

**HotelReceptionistAction** — 18 action types:

| Action Type | When to Use |
|---|---|
| `greet` | Opening of every interaction |
| `respond` | General conversational reply |
| `assign_room` | Check a guest in, assign a room number |
| `offer_upgrade` | Proactively offer a better room |
| `process_checkout` | Handle guest departure |
| `make_reservation` | Book a future stay |
| `offer_discount` | Apply a rate reduction |
| `offer_compensation` | Resolve complaints (free night, voucher) |
| `apologize` | De-escalate upset guests |
| `escalate_manager` | Hand off to management |
| `call_security` | Handle threatening situations |
| `call_maintenance` | Report room/facility issues |
| `arrange_transport` | Book taxi, airport shuttle, etc. |
| `order_room_service` | Process food/beverage requests |
| `provide_information` | Answer questions about hotel, area |
| `log_lost_item` | Record lost & found items |
| `transfer_call` | Route to another department |
| `end_interaction` | Close the episode politely |

Optional action fields: `room_number`, `discount_percent`, `compensation_details`, `reservation_details`, `service_details`, `lost_item_description`, `department`, `internal_notes`.

### Observation Space

**HotelReceptionistObservation** — per-step state:

| Field | Type | Description |
|---|---|---|
| `scenario_type` | str | Task category (e.g. `billing_dispute`, `vip_arrival`) |
| `scenario_difficulty` | int | 1–5 difficulty level |
| `guest_message` | str | Current guest utterance |
| `guest_profile` | dict | Name, mood, VIP status, loyalty tier, preferences |
| `hotel_state` | dict | Room availability by type with sample rooms + pricing |
| `conversation_history` | list | Last 8 turns `{role, message}` |
| `available_actions` | list | All 18 action types (always full set) |
| `reward_breakdown` | dict | Per-dimension scores from last step |
| `turn_number` / `max_turns` | int | Progress tracking (max = 10) |
| `done` | bool | Episode complete flag |
| `reward` | float | Step reward in [0.0, 1.0] |

### Scenario Types (15)

`check_in`, `check_out`, `reservation`, `complaint`, `room_service`, `concierge`, `lost_and_found`, `billing_dispute`, `emergency`, `vip_arrival`, `group_booking`, `noise_complaint`, `accessibility`, `late_checkout`, `loyalty_program`

### Reward Function

Each step is judged by a single LLM call (the "Unified Judge") that scores 4 dimensions and generates the guest's next reply:

| Dimension | Weight | What It Measures |
|---|---|---|
| Professionalism | 25% | Formal tone, hotel vocabulary, warmth |
| Accuracy | 35% | Correct action for the scenario, matches expected resolution |
| Empathy | 20% | Emotional intelligence, mood-appropriate response |
| Efficiency | 20% | Moving toward resolution without wasting turns |

**Reward pipeline:**
1. **Epsilon floor** — clamp each score to [0.05, 1.0] to preserve gradient signal
2. **Weighted geometric mean** — `p^0.25 × a^0.35 × e^0.20 × f^0.20` (punishes single weak dimension)
3. **Accuracy gate** — if accuracy < 0.2 → reward = 0.0, episode ends immediately
4. **Difficulty bonus** — quality amplification scales with difficulty (0–40% boost)
5. **Timeout penalty** — reward × 0.5 if episode reaches max turns without resolution

## Evaluation Tasks

The `inference.py` baseline runs 3 named tasks with fixed seeds for reproducibility:

| Task ID | Seed | Difficulty | Description |
|---|---|---|---|
| `easy_check_in` | 42 | 1/5 | Standard check-in for a happy guest |
| `medium_complaint` | 100 | 3/5 | Billing dispute from an upset guest |
| `hard_vip_emergency` | 200 | 5/5 | VIP guest with an emergency situation |

Score = mean reward across steps, clamped to [0.0, 1.0]. Task succeeds if score ≥ 0.4.

## Deploying to Hugging Face Spaces

```bash
cd hotel_receptionist
openenv push
```

After push, set these secrets in your Space settings:
- `HF_TOKEN` — your Hugging Face API token
- `MODEL_NAME` — `meta-llama/Llama-3.1-70B-Instruct` (recommended)
- `API_BASE_URL` — `https://router.huggingface.co/v1`

The deployed Space provides:
- **Web Interface** at `/web` — Interactive environment explorer
- **API Docs** at `/docs` — Full OpenAPI/Swagger interface
- **Health Check** at `/health` — `{"status": "ok"}`
- **WebSocket** at `/ws` — Low-latency persistent session endpoint

## Running Locally

```bash
cd hotel_receptionist

# Install dependencies
uv sync

# Load environment variables
export HF_TOKEN="hf_..."
export MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"

# Validate OpenEnv spec
openenv validate

# Start the server
uv run server

# Run inference in a separate terminal
python inference.py
```

## Project Structure

```
hotel_receptionist/
├── inference.py          ← baseline evaluation script (3 tasks, exact log format)
├── openenv.yaml          ← OpenEnv spec metadata
├── Dockerfile            ← multi-stage container build
├── README.md             ← this file
├── pyproject.toml        ← dependencies (openenv-core, httpx)
├── uv.lock               ← locked dependency versions
├── models.py             ← Pydantic Action + Observation models
├── client.py             ← HTTP/WebSocket client wrapper
├── __init__.py
└── server/
    ├── app.py            ← FastAPI server (step/reset/state endpoints, up to 10 sessions)
    ├── hotel_receptionist_environment.py  ← LLM-driven simulation + reward engine
    └── __init__.py
```
