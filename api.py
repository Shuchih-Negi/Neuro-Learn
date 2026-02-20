"""
api.py
======
FastAPI backend that exposes the attention state ML model to your React frontend.

Endpoints:
  POST /log-interaction   → logs a question event, triggers prediction
  GET  /attention-state/{user_id}  → returns latest state
  GET  /stability-history/{user_id} → returns time series for the graph
  GET  /parent-dashboard/{user_id}
  GET  /teacher-heatmap/{class_id}

Run:
  uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time
import collections
import numpy as np

# ── Lazy-load the predictor (uses rule-based fallback if no model saved) ──
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.feature_engineering import FEATURE_COLS, rule_based_state, STATE_TO_INT

# Try loading the trained LSTM; fall back to rule-based if not yet trained
try:
    from models.lstm_model import AttentionPredictor
    _predictor = AttentionPredictor("models/best_lstm.pt", "models/scaler.pkl")
    print("✅  LSTM model loaded")
    USE_LSTM = True
except Exception as e:
    print(f"⚠️  LSTM not found ({e}), using rule-based classifier")
    USE_LSTM = False

app = FastAPI(title="Adaptive Attention API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# IN-MEMORY STORES  (replace with your DB in production)
# ─────────────────────────────────────────────────────────────

# {user_id → deque of interaction dicts}
interaction_log:  Dict[int, collections.deque] = collections.defaultdict(
    lambda: collections.deque(maxlen=200)
)
# {user_id → latest attention result}
latest_state:     Dict[int, dict] = {}
# {user_id → list of {timestamp, score, state}}
stability_history: Dict[int, list] = collections.defaultdict(list)


# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────

class InteractionEvent(BaseModel):
    user_id:           int
    question_id:       int
    correct:           int            # 0 or 1
    response_time_s:   float          # seconds
    attempt_count:     int = 1
    hint_count:        int = 0
    hint_ratio:        float = 0.0
    overlap_time:      float = 0.0    # total time on question (seconds)
    # optional metadata
    skill_name:        Optional[str] = None
    session_id:        Optional[int] = None


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _compute_rolling_features(user_id: int, event: InteractionEvent) -> dict:
    """Build the feature dict expected by the predictor/rule engine."""
    history  = list(interaction_log[user_id])
    rt_vals  = [h["response_time_s"] for h in history] + [event.response_time_s]
    err_vals = [1 - h["correct"]     for h in history] + [1 - event.correct]

    rt_mean     = float(np.mean(rt_vals))
    rt_variance = float(np.var(rt_vals))
    rt_trend    = float(rt_vals[-1] - rt_vals[0]) if len(rt_vals) > 1 else 0.0
    error_rate  = float(np.mean(err_vals))
    error_burst = float(sum(err_vals[-3:]) / min(3, len(err_vals)))
    attempt_mean= float(np.mean([h["attempt_count"] for h in history] + [event.attempt_count]))
    hint_rate   = float(np.mean([h["hint_count"] for h in history] + [event.hint_count]))
    idle_s      = max(0.0, event.overlap_time - event.response_time_s)

    return {
        "rt_s":          event.response_time_s,
        "correct":       event.correct,
        "attempt_count": event.attempt_count,
        "hint_count":    event.hint_count,
        "hint_ratio":    event.hint_ratio,
        "idle_s":        idle_s,
        "rt_mean":       rt_mean,
        "rt_variance":   rt_variance,
        "rt_trend":      rt_trend,
        "error_rate":    error_rate,
        "error_burst":   error_burst,
        "attempt_mean":  attempt_mean,
        "hint_rate":     hint_rate,
        # derived for rule-based
        "_idle_s":       idle_s,
    }


def _rule_based_predict(features: dict) -> dict:
    """Fallback when LSTM is not loaded."""
    state = rule_based_state(
        rt_s          = features["rt_s"],
        correct       = features["correct"],
        attempt_count = features["attempt_count"],
        hint_count    = features["hint_count"],
        idle_s        = features["_idle_s"],
        rt_variance   = features["rt_variance"],
        error_burst   = features["error_burst"],
    )
    from models.lstm_model import _state_to_action
    # Import inline to avoid circular issues
    action_map = {
        "Focused":     "increase_difficulty",
        "Drifting":    "shorter_task",
        "Impulsive":   "add_scaffold",
        "Overwhelmed": "simplify_problem",
    }
    return {
        "state":         state,
        "confidence":    0.90,
        "probabilities": {s: (0.85 if s == state else 0.05) for s in ["Focused","Drifting","Impulsive","Overwhelmed"]},
        "action":        action_map[state],
    }


def _stability_score(rt_variance, error_burst, idle_s) -> float:
    norm_rt  = min(rt_variance / 200.0, 1.0)
    norm_err = min(error_burst, 1.0)
    norm_idl = min(idle_s / 30.0, 1.0)
    return max(0.0, round(100 * (1 - (0.30 * norm_rt + 0.40 * norm_err + 0.30 * norm_idl)), 1))


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.post("/log-interaction")
async def log_interaction(event: InteractionEvent):
    """
    Receives a question interaction event from the frontend,
    runs the attention model, stores the result, returns prediction.
    """
    features = _compute_rolling_features(event.user_id, event)

    if USE_LSTM:
        result = _predictor.push(event.user_id, features)
        if result["state"] is None:
            result = _rule_based_predict(features)
    else:
        result = _rule_based_predict(features)

    # Store interaction
    interaction_log[event.user_id].append({
        "response_time_s": event.response_time_s,
        "correct":         event.correct,
        "attempt_count":   event.attempt_count,
        "hint_count":      event.hint_count,
        "timestamp":       time.time(),
    })

    # Compute and store stability score
    score = _stability_score(
        rt_variance  = features["rt_variance"],
        error_burst  = features["error_burst"],
        idle_s       = features["idle_s"],
    )
    stability_history[event.user_id].append({
        "timestamp":  time.time(),
        "score":      score,
        "state":      result["state"],
    })

    latest_state[event.user_id] = {**result, "stability_score": score}

    return {
        "user_id":         event.user_id,
        "state":           result["state"],
        "confidence":      result["confidence"],
        "probabilities":   result["probabilities"],
        "action":          result["action"],
        "stability_score": score,
    }


@app.get("/attention-state/{user_id}")
async def get_attention_state(user_id: int):
    """Returns the latest attention prediction for a student."""
    if user_id not in latest_state:
        raise HTTPException(404, f"No data yet for user {user_id}")
    return latest_state[user_id]


@app.get("/stability-history/{user_id}")
async def get_stability_history(user_id: int, last_n: int = 50):
    """Returns time-series of stability scores for the graph widget."""
    history = stability_history.get(user_id, [])
    return {"user_id": user_id, "history": history[-last_n:]}


@app.get("/parent-dashboard/{user_id}")
async def parent_dashboard(user_id: int):
    """Aggregated analytics for the parent view."""
    history = stability_history.get(user_id, [])
    if not history:
        raise HTTPException(404, "No session data found")

    scores       = [h["score"] for h in history]
    states       = [h["state"] for h in history]
    state_counts = {s: states.count(s) for s in set(states)}
    total        = len(states)

    return {
        "user_id":          user_id,
        "avg_stability":    round(sum(scores) / len(scores), 1),
        "peak_stability":   max(scores),
        "lowest_stability": min(scores),
        "total_questions":  total,
        "state_distribution": {k: round(v / total, 2) for k, v in state_counts.items()},
        "impulsivity_index":  round(state_counts.get("Impulsive", 0) / total, 2),
        "overwhelm_frequency": round(state_counts.get("Overwhelmed", 0) / total, 2),
        "insight": _generate_insight(state_counts, scores),
    }


@app.get("/teacher-heatmap/{class_id}")
async def teacher_heatmap(class_id: int):
    """
    Aggregated class view. In production, filter by class_id.
    Here we just return stats across all tracked students.
    """
    class_summary = []
    for uid, history in stability_history.items():
        if not history:
            continue
        scores = [h["score"] for h in history]
        states = [h["state"] for h in history]
        class_summary.append({
            "user_id":       uid,
            "avg_stability": round(sum(scores) / len(scores), 1),
            "dominant_state": max(set(states), key=states.count),
        })

    if not class_summary:
        return {"class_id": class_id, "students": [], "alerts": []}

    # Intervention alerts
    alerts = []
    overwhelmed = [s for s in class_summary if s["dominant_state"] == "Overwhelmed"]
    impulsive   = [s for s in class_summary if s["dominant_state"] == "Impulsive"]
    if len(overwhelmed) >= 3:
        alerts.append(f"{len(overwhelmed)} students showing overload — consider simplifying.")
    if len(impulsive) >= 3:
        alerts.append(f"{len(impulsive)} students showing high impulsivity — slow the pace.")

    return {
        "class_id":   class_id,
        "students":   class_summary,
        "alerts":     alerts,
        "avg_class_stability": round(
            sum(s["avg_stability"] for s in class_summary) / len(class_summary), 1
        ),
    }


def _generate_insight(state_counts: dict, scores: list) -> str:
    dominant = max(state_counts, key=state_counts.get)
    avg      = sum(scores) / len(scores)
    if dominant == "Impulsive":
        return "Your child tends to answer quickly without checking — try untimed practice."
    if dominant == "Overwhelmed":
        return "Your child struggles with cognitive load — shorter sessions may help."
    if dominant == "Drifting":
        return "Focus drifts mid-session — try sessions between 4–6 PM when energy is higher."
    if avg > 75:
        return "Great focus stability this week — keep the current routine."
    return "Steady progress — try increasing problem variety to maintain engagement."


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK + DEMO MODE INFO
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status":     "running",
        "model_mode": "LSTM" if USE_LSTM else "rule-based",
        "endpoints":  [
            "POST /log-interaction",
            "GET  /attention-state/{user_id}",
            "GET  /stability-history/{user_id}",
            "GET  /parent-dashboard/{user_id}",
            "GET  /teacher-heatmap/{class_id}",
        ],
    }
