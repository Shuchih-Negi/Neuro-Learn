"""
feature_engineering.py
=======================
Transforms raw interaction logs into per-question feature vectors
using a rolling window over the last N questions per student.

Input columns required:
    user_id, order_id, correct, attempt_count,
    response_time_s, hint_count, hint_ratio, overlap_time

Output:
    One row per question with engineered features + attention label
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

WINDOW = 5          # rolling window size (last N questions)
MIN_ROWS = 3        # minimum history before we emit a feature row


# ─────────────────────────────────────────────────────────────
# RULE-BASED LABELER (used when no ground-truth label exists)
# ─────────────────────────────────────────────────────────────

def rule_based_state(
    rt_s: float,
    correct: int,
    attempt_count: int,
    hint_count: int,
    idle_s: float,
    rt_variance: float,
    error_burst: float,
) -> str:
    """
    Deterministic classifier — Phase 1 of the spec.
    Returns one of: Focused | Drifting | Impulsive | Overwhelmed

    Priority order matters (most specific first).
    """
    # Impulsive: answered blazingly fast AND got it wrong, no hints
    if rt_s < 3.5 and correct == 0 and hint_count == 0 and attempt_count <= 1:
        return "Impulsive"

    # Overwhelmed: many attempts or many hints with poor accuracy
    if attempt_count >= 3 or hint_count >= 2:
        return "Overwhelmed"

    # Drifting: long idle / slow RT
    if idle_s > 12 or rt_s > 30:
        return "Drifting"

    # High RT variance in recent window → drifting
    if rt_variance > 60:
        return "Drifting"

    # High error burst → either impulsive or overwhelmed
    if error_burst > 0.6:
        return "Overwhelmed" if hint_count > 0 else "Impulsive"

    return "Focused"


# ─────────────────────────────────────────────────────────────
# PER-STUDENT ROLLING FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────

def _extract_student_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Given all rows for a single student (sorted by order_id),
    compute rolling window features for each row.
    """
    group = group.sort_values("order_id").reset_index(drop=True)
    records = []

    for i in range(len(group)):
        if i < MIN_ROWS - 1:
            continue   # not enough history yet

        window_start = max(0, i - WINDOW + 1)
        w = group.iloc[window_start : i + 1]   # rolling window
        current = group.iloc[i]                # current question

        rt_vals     = w["response_time_s"].values
        correct_vals= w["correct"].values
        attempt_vals= w["attempt_count"].values
        hint_vals   = w["hint_count"].values

        # ── Rolling statistics ────────────────────────────────
        rt_mean      = float(np.mean(rt_vals))
        rt_variance  = float(np.var(rt_vals))
        rt_trend     = float(rt_vals[-1] - rt_vals[0]) if len(rt_vals) > 1 else 0.0

        error_rate   = float(1 - np.mean(correct_vals))
        error_burst  = float(np.sum(correct_vals[-3:] == 0) / min(3, len(correct_vals)))
        attempt_mean = float(np.mean(attempt_vals))
        hint_rate    = float(np.mean(hint_vals))
        idle_s       = float(current.get("overlap_time", rt_mean) - current["response_time_s"])
        idle_s       = max(0.0, idle_s)

        # ── Attention Stability Score (0–100) ─────────────────
        # Mirrors the formula in the spec
        norm_rt_var  = min(rt_variance / 200.0, 1.0)
        norm_error   = min(error_burst, 1.0)
        norm_idle    = min(idle_s / 30.0, 1.0)
        stability    = max(0.0, 100 * (1 - (
            0.30 * norm_rt_var +
            0.40 * norm_error  +
            0.30 * norm_idle
        )))

        # ── Rule-based label (used if no true_state column) ───
        auto_label = rule_based_state(
            rt_s        = float(current["response_time_s"]),
            correct     = int(current["correct"]),
            attempt_count = int(current["attempt_count"]),
            hint_count  = int(current["hint_count"]),
            idle_s      = idle_s,
            rt_variance = rt_variance,
            error_burst = error_burst,
        )

        rec = {
            # ── identifiers
            "user_id":          current["user_id"],
            "order_id":         current["order_id"],
            # ── current-question features
            "rt_s":             float(current["response_time_s"]),
            "correct":          int(current["correct"]),
            "attempt_count":    int(current["attempt_count"]),
            "hint_count":       int(current["hint_count"]),
            "hint_ratio":       float(current.get("hint_ratio", 0)),
            "idle_s":           idle_s,
            # ── rolling-window features
            "rt_mean":          rt_mean,
            "rt_variance":      rt_variance,
            "rt_trend":         rt_trend,
            "error_rate":       error_rate,
            "error_burst":      error_burst,
            "attempt_mean":     attempt_mean,
            "hint_rate":        hint_rate,
            # ── composite score
            "stability_score":  stability,
            # ── label
            "label_auto":       auto_label,
        }

        # Keep ground truth if the synthetic dataset provided it
        if "true_state" in current:
            rec["label_true"] = current["true_state"]

        records.append(rec)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

STATE_TO_INT = {
    "Focused":     0,
    "Drifting":    1,
    "Impulsive":   2,
    "Overwhelmed": 3,
}
INT_TO_STATE = {v: k for k, v in STATE_TO_INT.items()}


def build_features(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on a raw interaction log.

    Parameters
    ----------
    df        : raw dataframe (output of data_loader)
    label_col : 'label_true' to use synthetic ground-truth,
                'label_auto' to use rule-based labeling (default)

    Returns
    -------
    feature_df with numeric label column 'y'
    """
    print("Building features per student…")
    parts = []
    for uid, group in df.groupby("user_id"):
        parts.append(_extract_student_features(group))

    feat_df = pd.concat(parts, ignore_index=True)

    # Decide which label column to encode as 'y'
    if label_col is None:
        label_col = "label_true" if "label_true" in feat_df.columns else "label_auto"

    feat_df["y"] = feat_df[label_col].map(STATE_TO_INT)
    print(f"Feature matrix: {feat_df.shape}  |  label source: '{label_col}'")
    print("Class distribution:\n", feat_df[label_col].value_counts())
    return feat_df


# ─────────────────────────────────────────────────────────────
# FEATURE COLUMNS (used downstream by model training)
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "rt_s",
    "correct",
    "attempt_count",
    "hint_count",
    "hint_ratio",
    "idle_s",
    "rt_mean",
    "rt_variance",
    "rt_trend",
    "error_rate",
    "error_burst",
    "attempt_mean",
    "hint_rate",
]


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from utils.data_loader import generate_synthetic_dataset

    raw = generate_synthetic_dataset(n_students=20)
    feat = build_features(raw, label_col="label_true")
    print(feat[FEATURE_COLS + ["stability_score", "label_true", "y"]].head(10))
