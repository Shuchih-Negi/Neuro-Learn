"""
ml/attention_model.py
=====================
Attention state prediction layer.

Two-stage design:
    Stage 1  —  Per-modality prediction
        • EyeTracker        → eye-based state  (live, continuous)
        • LSTMPredictor     → behaviour-based  (after each question answer)
          └── falls back to RuleBasedPredictor if model file not present

    Stage 2  —  Kalman-inspired fusion
        FusionEngine combines both signals using confidence-weighted voting
        with exponential smoothing so the state doesn't thrash between
        questions.

Public API
----------
    model = AttentionModel(lstm_path=None)   # pass path to use real LSTM
    model.update(
        response_time_s = rt,
        correct         = True/False,
        eye_result      = eye_thread.get_latest(),   # or None
    )
    state = model.state        # str
    conf  = model.confidence   # float 0-1
    feats = model.last_features  # dict (useful for logging / LSTM training)
"""

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

STATES = ["Focused", "Drifting", "Impulsive", "Overwhelmed"]


# ─────────────────────────────────────────────────────────────
# Rule-Based Predictor  (mirrors the LSTM's feature space)
# ─────────────────────────────────────────────────────────────

class RuleBasedPredictor:
    """
    Heuristic classifier using the same 13 features the LSTM was trained on.
    Used as a drop-in fallback and also as a sanity-check alongside the LSTM.
    """

    def __init__(self):
        self._history: List[Dict] = []

    def predict(self, features: Dict) -> Tuple[str, float]:
        """
        Returns (state_str, confidence).
        Requires at least 3 historical samples for variance calculation.
        """
        self._history.append(features)
        h = self._history

        if len(h) < 3:
            return "Focused", 0.5

        recent = h[-5:]
        rts    = [r["rt_s"] for r in recent]
        rt_var = float(np.var(rts))
        err_burst = sum(1 for r in h[-3:] if not r["correct"]) / 3.0

        rt     = features["rt_s"]
        idle   = features.get("idle_s", 0.0)
        correct = bool(features["correct"])

        # Priority-ordered rules (most specific first)
        if err_burst >= 0.67 and rt < 4.0:
            return "Impulsive", 0.82

        if idle > 12.0 or (rt_var > 35.0 and not correct):
            return "Drifting", 0.78

        if not correct and rt > 22.0:
            return "Overwhelmed", 0.80

        if correct and rt < 8.0 and rt_var < 10.0:
            return "Focused", 0.90

        return "Focused", 0.70

    def reset(self):
        self._history.clear()


# ─────────────────────────────────────────────────────────────
# LSTM Predictor stub
# ─────────────────────────────────────────────────────────────

class LSTMPredictor:
    """
    Wrapper around the trained LSTM model.

    If the model file exists, loads and runs it.
    If not, raises FileNotFoundError so AttentionModel falls back gracefully.

    Expected model interface (from models/lstm_model.py):
        predictor = AttentionPredictor(model_path, scaler_path)
        result    = predictor.push(user_id=1, feature_dict=features)
        # result = {"state": str, "confidence": float}
    """

    FEATURE_ORDER = [
        "rt_s", "correct", "attempt_count", "hint_count", "hint_ratio",
        "idle_s", "rt_mean", "rt_variance", "rt_trend",
        "error_rate", "error_burst", "attempt_mean", "hint_rate",
    ]

    def __init__(self, model_path: str, scaler_path: str):
        # Lazy import so the file can be used even without torch installed
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from models.lstm_model import AttentionPredictor  # type: ignore
            self._inner = AttentionPredictor(model_path, scaler_path)
            print("[LSTM] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"LSTM load failed: {e}") from e

    def predict(self, features: Dict, user_id: int = 1) -> Tuple[str, float]:
        result = self._inner.push(user_id=user_id, feature_dict=features)
        return result["state"], result.get("confidence", 0.75)


# ─────────────────────────────────────────────────────────────
# Fusion Engine
# ─────────────────────────────────────────────────────────────

class FusionEngine:
    """
    Combines behaviour-based ML prediction and eye-tracking prediction.

    Strategy
    --------
    1. Start with the ML (behaviour) prediction as the base.
    2. If the eye tracker has a high-confidence reading (> EYE_CONF_THRESHOLD),
       it gets a weighted vote.
    3. Apply exponential smoothing so consecutive one-off detections don't
       flip the state instantly — this prevents jitter between questions.
    4. The smoothed probability vector is argmax-ed to get the final state.

    Parameters
    ----------
    smoothing : float
        Exponential smoothing factor α. Higher = reacts faster (0.0–1.0).
    eye_conf_threshold : float
        Minimum eye-tracker confidence before it influences the vote.
    eye_weight : float
        Proportion of the vote given to eye tracker when it is confident.
    """

    def __init__(
        self,
        smoothing:          float = 0.5,
        eye_conf_threshold: float = 0.55,
        eye_weight:         float = 0.4,
    ):
        self._alpha     = smoothing
        self._eye_thr   = eye_conf_threshold
        self._eye_w     = eye_weight
        # Smoothed probability distribution over STATES
        self._probs = np.array([1.0, 0.0, 0.0, 0.0])   # start Focused

    def fuse(
        self,
        ml_state:  str,
        ml_conf:   float,
        eye_state: Optional[str],
        eye_conf:  float,
    ) -> Tuple[str, float]:
        """
        Returns (final_state, final_confidence).
        """
        # Build current observation vector
        ml_vec  = self._one_hot(ml_state)  * ml_conf
        obs_vec = ml_vec.copy()

        if eye_state is not None and eye_conf >= self._eye_thr:
            eye_vec  = self._one_hot(eye_state) * eye_conf
            ml_w     = 1.0 - self._eye_w
            obs_vec  = ml_w * ml_vec + self._eye_w * eye_vec

        # Normalise
        total = obs_vec.sum()
        if total > 0:
            obs_vec /= total

        # Exponential smoothing
        self._probs = self._alpha * obs_vec + (1.0 - self._alpha) * self._probs

        idx   = int(np.argmax(self._probs))
        state = STATES[idx]
        conf  = float(self._probs[idx])
        return state, conf

    def reset(self):
        self._probs = np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _one_hot(state: str) -> np.ndarray:
        vec = np.zeros(len(STATES))
        if state in STATES:
            vec[STATES.index(state)] = 1.0
        return vec


# ─────────────────────────────────────────────────────────────
# AttentionModel — main public interface
# ─────────────────────────────────────────────────────────────

class AttentionModel:
    """
    Top-level attention model used by main.py.

    Usage
    -----
        model = AttentionModel(lstm_path="models/best_lstm.pt",
                               scaler_path="models/scaler.pkl")
        # after each question answer:
        model.update(response_time_s=rt, correct=True, eye_result=snap)
        print(model.state, model.confidence)
    """

    def __init__(
        self,
        lstm_path:   Optional[str] = None,
        scaler_path: Optional[str] = None,
    ):
        # Try LSTM first, fall back to rule-based
        self._lstm: Optional[LSTMPredictor] = None
        if lstm_path and Path(lstm_path).exists():
            try:
                self._lstm = LSTMPredictor(lstm_path, scaler_path or "")
                self._ml_type = "LSTM"
            except Exception as e:
                print(f"[AttentionModel] LSTM unavailable ({e}), using rule-based.")

        if self._lstm is None:
            self._rule = RuleBasedPredictor()
            self._ml_type = "Rule-based"
        else:
            self._rule = None

        self._fusion = FusionEngine(smoothing=0.5, eye_conf_threshold=0.55, eye_weight=0.4)

        # Running session stats
        self._times:     List[float] = []
        self._correct:   List[bool]  = []
        self._attempt    = 0

        # Public state
        self.state         = "Focused"
        self.confidence    = 0.5
        self.last_features: Dict = {}

    # ── public ──────────────────────────────────────────────

    def update(
        self,
        response_time_s: float,
        correct:         bool,
        eye_result:      Optional[Dict] = None,
    ) -> Tuple[str, float]:
        """
        Call once after each question answer.
        Updates self.state, self.confidence, self.last_features.
        Returns (state, confidence).
        """
        self._attempt   += 1
        self._times.append(response_time_s)
        self._correct.append(correct)

        features = self._build_features(response_time_s, correct)
        self.last_features = features

        # ML prediction
        if self._lstm:
            ml_state, ml_conf = self._lstm.predict(features)
        else:
            ml_state, ml_conf = self._rule.predict(features)

        # Eye prediction
        eye_state = eye_conf = None
        if eye_result:
            eye_state = eye_result.get("attention_state")
            eye_conf  = eye_result.get("confidence", 0.0)

        # Fuse
        self.state, self.confidence = self._fusion.fuse(
            ml_state, ml_conf, eye_state, eye_conf
        )
        return self.state, self.confidence

    @property
    def ml_type(self) -> str:
        return self._ml_type

    def reset(self):
        """Call between sessions."""
        self._times.clear()
        self._correct.clear()
        self._attempt = 0
        self.state    = "Focused"
        self.confidence = 0.5
        self._fusion.reset()
        if self._rule:
            self._rule.reset()

    # ── private ─────────────────────────────────────────────

    def _build_features(self, rt: float, correct: bool) -> Dict:
        """Build the 13-feature dict that matches LSTM training schema."""
        h     = len(self._times)
        rts   = self._times
        avg   = float(np.mean(rts)) if rts else rt
        var   = float(np.var(rts))  if len(rts) > 1 else 0.0
        trend = float(rts[-1] - rts[-2]) if len(rts) >= 2 else 0.0

        recent_corr  = self._correct[-5:] if len(self._correct) >= 5 else self._correct
        error_rate   = 1.0 - (sum(recent_corr) / len(recent_corr)) if recent_corr else 0.0
        err_burst    = sum(1 for c in self._correct[-3:] if not c) / 3.0 if len(self._correct) >= 3 else 0.0

        return {
            "rt_s":          rt,
            "correct":       int(correct),
            "attempt_count": self._attempt,
            "hint_count":    0,
            "hint_ratio":    0.0,
            "idle_s":        max(0.0, rt - 5.0),
            "rt_mean":       avg,
            "rt_variance":   var,
            "rt_trend":      trend,
            "error_rate":    error_rate,
            "error_burst":   err_burst,
            "attempt_mean":  self._attempt,
            "hint_rate":     0.0,
        }
