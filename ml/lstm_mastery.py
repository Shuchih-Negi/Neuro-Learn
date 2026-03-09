"""
ml/lstm_mastery.py  — NeuroLearn Language Edition
===================================================
LSTM-based Knowledge Tracing for language skill mastery.

Architecture (Section 6.2 of blueprint):
    Embedding Layer  → maps (skill_id, exercise_type) to dense vectors
    LSTM (2 layers, 128 hidden) → temporal sequence modelling
    Dense → Sigmoid → mastery probability per skill

Research basis
──────────────
• Piech et al. (2015) — Deep Knowledge Tracing:
    LSTM over interaction sequences outperforms Bayesian KT for predicting
    future correctness. Applied here to language skill sequences.

• Corbett & Anderson (1994) — Bayesian Knowledge Tracing:
    P(mastery | evidence) updated per interaction. We use this as a warm-start
    prior before the LSTM accumulates enough sequence data (< 10 interactions).

• Ebbinghaus (1885) — Forgetting Curve:
    Mastery decays without review. The time_since_last_review feature
    captures this directly in the LSTM input.

• Sison & Shimura (1998) — Student Modelling in ITS:
    Multi-skill tracking is necessary — learners have heterogeneous mastery
    profiles across vocabulary / grammar / listening etc.

Input features per interaction (extends blueprint spec with language features)
──────────────────────────────────────────────────────────────────────────────
    skill_id              — embedded (vocab/grammar/listening/etc.)
    exercise_type_id      — embedded (mcq/fill/translation/etc.)
    correctness           — 0/1
    time_taken_s          — response latency
    hint_used             — 0/1
    attention_score       — 0-1 from AttentionModel
    error_type_encoded    — 0=none, 1=vocab, 2=grammar, 3=pronunciation, 4=word_order
    time_since_review_s   — seconds since last practice of this skill (forgetting)
    attempt_count         — cumulative attempts on this skill
    session_position      — which question number in session (fatigue proxy)

Output
──────
    mastery_probability   — float 0-1 per skill (sigmoid output)

Training
────────
    See ml_training/train_lstm.py for data pipeline.
    Requires: interactions table (exported as CSV) with columns above.
    Loss: Binary Cross Entropy on next-question correctness.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy torch import so module loads without GPU/torch installed
_torch_available = False
try:
    import torch
    import torch.nn as nn
    _torch_available = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────
# Skill & Exercise type catalogues (must match moderator.py)
# ─────────────────────────────────────────────────────────────

SKILL_IDS = {
    "vocabulary_basic":        0,
    "vocabulary_intermediate": 1,
    "grammar_present":         2,
    "grammar_past":            3,
    "grammar_future":          4,
    "grammar_genders":         5,
    "grammar_plurals":         6,
    "grammar_adjectives":      7,
    "grammar_pronouns":        8,
    "reading_comprehension":   9,
    "translation_basic":       10,
    "translation_intermediate": 11,
    "listening_comprehension": 12,
    "sentence_structure":      13,
    "numbers_time":            14,
    "social_phrases":          15,
}

EXERCISE_IDS = {
    "multiple_choice_vocab": 0,
    "fill_in_the_blank":     1,
    "translation":           2,
    "grammar_sort":          3,
    "listening_text":        4,
    "match_pairs":           5,
    "sentence_builder":      6,
}

ERROR_TYPES = {
    "none":         0,
    "vocabulary":   1,
    "grammar":      2,
    "pronunciation":3,
    "word_order":   4,
    "spelling":     5,
}

N_SKILLS     = len(SKILL_IDS)
N_EX_TYPES   = len(EXERCISE_IDS)
N_ERRORS     = len(ERROR_TYPES)
EMBED_DIM    = 16
NUMERIC_FEATS = 6   # correctness, time_taken_s, hint_used, attention_score,
                    # time_since_review_s (normalised), session_position (normalised)
INPUT_DIM    = EMBED_DIM * 2 + EMBED_DIM + NUMERIC_FEATS  # skill + ex_type + error + numeric


# ─────────────────────────────────────────────────────────────
# PyTorch Model Definition
# ─────────────────────────────────────────────────────────────

if _torch_available:
    class LSTMMasteryModel(nn.Module):
        """
        Deep Knowledge Tracing LSTM for language skill mastery.
        Processes interaction sequences and outputs P(mastery) per step.
        """

        def __init__(
            self,
            n_skills:    int = N_SKILLS,
            n_ex_types:  int = N_EX_TYPES,
            n_errors:    int = N_ERRORS,
            embed_dim:   int = EMBED_DIM,
            numeric_feats: int = NUMERIC_FEATS,
            hidden_size: int = 128,
            num_layers:  int = 2,
            dropout:     float = 0.3,
        ):
            super().__init__()
            self.skill_embed   = nn.Embedding(n_skills,   embed_dim)
            self.extype_embed  = nn.Embedding(n_ex_types, embed_dim)
            self.error_embed   = nn.Embedding(n_errors,   embed_dim)

            input_dim = embed_dim * 3 + numeric_feats
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.head    = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, skill_ids, extype_ids, error_ids, numeric, hidden=None):
            """
            Args:
                skill_ids:  (batch, seq_len) int
                extype_ids: (batch, seq_len) int
                error_ids:  (batch, seq_len) int
                numeric:    (batch, seq_len, numeric_feats) float
                hidden:     optional (h_n, c_n) for incremental inference

            Returns:
                mastery:    (batch, seq_len, 1) float  — P(mastery) at each step
                hidden:     (h_n, c_n)                — carry for next call
            """
            s = self.skill_embed(skill_ids)    # (B, T, E)
            e = self.extype_embed(extype_ids)   # (B, T, E)
            r = self.error_embed(error_ids)     # (B, T, E)
            x = torch.cat([s, e, r, numeric], dim=-1)  # (B, T, 3E+N)
            out, hidden = self.lstm(x, hidden)
            out = self.dropout(out)
            mastery = self.head(out)            # (B, T, 1)
            return mastery, hidden


# ─────────────────────────────────────────────────────────────
# Inference wrapper
# ─────────────────────────────────────────────────────────────

class LSTMMasteryPredictor:
    """
    Online mastery predictor. Maintains per-user LSTM hidden state
    so each new answer updates the mastery estimate incrementally.

    Usage
    -----
        predictor = LSTMMasteryPredictor("models/lstm_mastery.pt")
        result = predictor.push(user_id="u1", interaction={...})
        print(result["mastery_probability"])   # 0.0–1.0
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model = None
        self._hidden: Dict[str, Tuple] = {}   # user_id → (h_n, c_n)
        self._loaded = False

        if model_path and _torch_available and Path(model_path).exists():
            try:
                self._model = LSTMMasteryModel()
                self._model.load_state_dict(
                    torch.load(model_path, map_location="cpu")
                )
                self._model.eval()
                self._loaded = True
                print(f"[LSTMMasteryPredictor] Model loaded from {model_path}")
            except Exception as e:
                print(f"[LSTMMasteryPredictor] Load failed ({e}), using EWMA fallback.")

    def push(self, user_id: str, interaction: Dict) -> Dict:
        """
        Process one interaction and return updated mastery estimate.

        interaction keys:
            skill_tag         str
            exercise_type     str
            correctness       float (0 or 1)
            time_taken_s      float
            hint_used         bool
            attention_score   float
            error_type        str
            time_since_review_s float
            session_position  int   (1-based)
            total_questions   int
        """
        if self._loaded and self._model is not None:
            return self._lstm_push(user_id, interaction)
        return self._ewma_push(user_id, interaction)

    def reset_user(self, user_id: str):
        """Call at session start to clear LSTM hidden state."""
        self._hidden.pop(user_id, None)

    # ── LSTM path ────────────────────────────────────────────

    def _lstm_push(self, user_id: str, interaction: Dict) -> Dict:
        import torch
        skill_id  = SKILL_IDS.get(interaction.get("skill_tag", ""), 0)
        extype_id = EXERCISE_IDS.get(interaction.get("exercise_type", ""), 0)
        error_id  = ERROR_TYPES.get(interaction.get("error_type", "none"), 0)

        numeric = self._build_numeric(interaction)
        skill_t  = torch.tensor([[[skill_id]]], dtype=torch.long)   # (1,1)
        extype_t = torch.tensor([[[extype_id]]], dtype=torch.long)
        error_t  = torch.tensor([[[error_id]]], dtype=torch.long)
        num_t    = torch.tensor([[numeric]], dtype=torch.float32)    # (1,1,N)

        # Remove extra dim
        skill_t  = skill_t.squeeze(-1)
        extype_t = extype_t.squeeze(-1)
        error_t  = error_t.squeeze(-1)

        hidden = self._hidden.get(user_id)
        with torch.no_grad():
            mastery_t, new_hidden = self._model(
                skill_t, extype_t, error_t, num_t, hidden
            )
        self._hidden[user_id] = new_hidden
        mastery = float(mastery_t.squeeze().item())
        return self._make_result(mastery, interaction, source="lstm")

    # ── EWMA fallback (no torch) ─────────────────────────────

    def _ewma_push(self, user_id: str, interaction: Dict) -> Dict:
        """
        Lightweight EWMA approximation used when LSTM model not loaded.
        Bayesian KT-inspired: P(mastery) = EWMA of correctness weighted
        by attention_score (high attention → higher weight).
        """
        if not hasattr(self, "_ewma_state"):
            self._ewma_state: Dict = {}

        key = f"{user_id}:{interaction.get('skill_tag','')}"
        state = self._ewma_state.get(key, {"mastery": 0.0, "count": 0})

        performance = float(interaction.get("correctness", 0))
        attention   = float(interaction.get("attention_score", 0.5))
        # Attention-weighted EWMA — focused learning has higher α
        alpha = 0.25 + 0.15 * attention   # 0.25–0.40
        state["mastery"] = (1 - alpha) * state["mastery"] + alpha * performance
        state["count"]  += 1
        self._ewma_state[key] = state

        return self._make_result(state["mastery"], interaction, source="ewma")

    # ── Shared ───────────────────────────────────────────────

    @staticmethod
    def _build_numeric(ix: Dict) -> List[float]:
        pos = ix.get("session_position", 1)
        tot = max(1, ix.get("total_questions", 10))
        return [
            float(ix.get("correctness", 0)),
            min(1.0, float(ix.get("time_taken_s", 10)) / 60.0),   # normalise to 60s
            float(ix.get("hint_used", 0)),
            float(ix.get("attention_score", 0.5)),
            min(1.0, float(ix.get("time_since_review_s", 0)) / (7 * 86400)),  # norm to 7d
            (pos - 1) / (tot - 1) if tot > 1 else 0.0,            # session progress 0-1
        ]

    @staticmethod
    def _make_result(mastery: float, interaction: Dict, source: str) -> Dict:
        skill = interaction.get("skill_tag", "unknown")
        # Map mastery to Krashen difficulty recommendation
        if mastery < 0.35:
            recommended_action = "reinforce"      # i-1: more of the same
        elif mastery < 0.65:
            recommended_action = "extend"         # i: varied contexts
        else:
            recommended_action = "advance"        # i+1: next skill
        return {
            "skill_tag":          skill,
            "mastery_probability": round(mastery, 4),
            "recommended_action": recommended_action,
            "source":             source,
        }


# ─────────────────────────────────────────────────────────────
# Convenience: batch mastery snapshot from stored interactions
# ─────────────────────────────────────────────────────────────

def mastery_snapshot_from_history(
    history: List[Dict],
    predictor: Optional[LSTMMasteryPredictor] = None,
) -> Dict[str, float]:
    """
    Replay a history list to compute mastery per skill.
    Useful for dashboard and LSTM training data generation.

    Args:
        history: list of interaction dicts (from progress_store)
        predictor: optional pre-loaded predictor (creates new EWMA if None)
    Returns:
        {skill_tag: mastery_probability}
    """
    if predictor is None:
        predictor = LSTMMasteryPredictor()   # EWMA fallback

    user_id = "snapshot"
    predictor.reset_user(user_id)
    result = {}
    for ix in history:
        r = predictor.push(user_id, ix)
        result[r["skill_tag"]] = r["mastery_probability"]
    return result
