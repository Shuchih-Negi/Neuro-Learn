"""
data_loader.py
==============
Loads the ASSISTments 2009-2010 dataset (or simulates one with the same schema).

ASSISTments columns we care about:
  - order_id          : global sequence order
  - user_id           : student ID
  - problem_id        : question ID
  - correct           : 1 = correct, 0 = incorrect
  - attempt_count     : number of attempts on this problem
  - ms_first_response : time (ms) before first response — our proxy for response_time
  - hint_count        : hints requested
  - hint_total        : total hints available
  - overlap_time      : seconds student spent on problem

Download the real dataset from:
  https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
  File: skill_builder_data_corrected.csv

If not available, run generate_synthetic_dataset() to get a hackathon-ready stand-in.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# 1.  LOAD REAL ASSISTMENTS DATA
# ─────────────────────────────────────────────

def load_assistments(path: str) -> pd.DataFrame:
    """
    Load and normalise the ASSISTments CSV.
    Returns a clean DataFrame ready for feature engineering.
    """
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    rename = {
        "order_id":          "order_id",
        "user_id":           "user_id",
        "problem_id":        "problem_id",
        "correct":           "correct",
        "attempt_count":     "attempt_count",
        "ms_first_response": "ms_first_response",
        "hint_count":        "hint_count",
        "hint_total":        "hint_total",
        "overlap_time":      "overlap_time",
        "skill_name":        "skill_name",
    }
    # Keep only columns that actually exist in the file
    existing = {k: v for k, v in rename.items() if k in df.columns}
    df = df[list(existing.keys())].rename(columns=existing)

    # Derived columns
    df["response_time_s"] = df["ms_first_response"] / 1000.0   # ms → seconds
    df["hint_ratio"]      = df["hint_count"] / (df["hint_total"].replace(0, 1))
    df["correct"]         = df["correct"].astype(int)

    # Drop rows with obviously corrupt response times
    df = df[(df["response_time_s"] > 0) & (df["response_time_s"] < 600)]
    df = df.sort_values(["user_id", "order_id"]).reset_index(drop=True)

    print(f"Loaded ASSISTments: {len(df):,} rows, {df['user_id'].nunique():,} students")
    return df


# ─────────────────────────────────────────────
# 2.  SYNTHETIC DATASET (hackathon fallback)
# ─────────────────────────────────────────────

def generate_synthetic_dataset(
    n_students: int = 300,
    sessions_per_student: int = 5,
    questions_per_session: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset that perfectly mirrors the ASSISTments schema
    but also embeds explicit attention state patterns so the model has something
    real to learn.

    Attention regimes per student (randomly assigned each session):
      Focused     → fast RT, high accuracy, low hints
      Drifting    → slow RT, medium accuracy, idle spikes
      Impulsive   → very fast RT, low accuracy, no hints
      Overwhelmed → high retries, many hints, slow RT
    """
    rng = np.random.default_rng(seed)
    STATES = ["Focused", "Drifting", "Impulsive", "Overwhelmed"]

    rows = []
    order_counter = 0

    for uid in range(n_students):
        for sid in range(sessions_per_student):
            # Each session has a dominant state that shifts mid-session
            primary   = rng.choice(STATES)
            secondary = rng.choice(STATES)
            switch_at = rng.integers(8, 15)   # switch state after this question

            for qidx in range(questions_per_session):
                state = primary if qidx < switch_at else secondary

                # ── Per-state signal distributions ──────────────────────────
                if state == "Focused":
                    rt_s       = max(1.0, rng.normal(8,  2))
                    correct    = int(rng.random() < 0.85)
                    attempts   = rng.choice([1, 1, 1, 2], p=[.7,.1,.1,.1])
                    hints      = rng.choice([0, 0, 1],    p=[.8,.1,.1])
                    idle_s     = max(0, rng.normal(2, 1))

                elif state == "Drifting":
                    rt_s       = max(3.0, rng.normal(25, 8))
                    correct    = int(rng.random() < 0.55)
                    attempts   = rng.choice([1, 2, 3],    p=[.5,.3,.2])
                    hints      = rng.choice([0, 1, 2],    p=[.5,.3,.2])
                    idle_s     = max(0, rng.normal(12, 5))

                elif state == "Impulsive":
                    rt_s       = max(0.5, rng.normal(2.5, 0.8))
                    correct    = int(rng.random() < 0.30)
                    attempts   = rng.choice([1, 1, 2],    p=[.6,.2,.2])
                    hints      = rng.choice([0, 0, 0, 1], p=[.7,.1,.1,.1])
                    idle_s     = max(0, rng.normal(1, 0.5))

                else:  # Overwhelmed
                    rt_s       = max(5.0, rng.normal(35, 10))
                    correct    = int(rng.random() < 0.25)
                    attempts   = rng.choice([2, 3, 4, 5], p=[.3,.3,.2,.2])
                    hints      = rng.choice([1, 2, 3],    p=[.4,.3,.3])
                    idle_s     = max(0, rng.normal(8, 3))

                rows.append({
                    "order_id":          order_counter,
                    "user_id":           uid,
                    "session_id":        sid,
                    "problem_id":        rng.integers(1000, 5000),
                    "correct":           correct,
                    "attempt_count":     attempts,
                    "ms_first_response": rt_s * 1000,
                    "hint_count":        hints,
                    "hint_total":        3,
                    "overlap_time":      rt_s + idle_s,
                    "skill_name":        rng.choice(
                        ["Fractions","Algebra","Geometry","Decimals","Ratios"]
                    ),
                    "true_state":        state,   # ground truth label ✓
                })
                order_counter += 1

    df = pd.DataFrame(rows)
    df["response_time_s"] = df["ms_first_response"] / 1000.0
    df["hint_ratio"]      = df["hint_count"] / df["hint_total"]

    print(f"Generated synthetic dataset: {len(df):,} rows, "
          f"{df['user_id'].nunique()} students")
    print("State distribution:\n", df["true_state"].value_counts())
    return df


# ─────────────────────────────────────────────
# 3.  QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_synthetic_dataset(n_students=50)
    print(df.head())
    print(df.dtypes)
