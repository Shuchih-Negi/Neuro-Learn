"""
ml_training/train_lstm.py  — NeuroLearn Language Edition
=========================================================
Training pipeline for the LSTM Knowledge Tracing model.

Research: Piech et al. (2015) — Deep Knowledge Tracing
  "Knowledge Tracing: Modelling the Acquisition of Procedural Knowledge"
  NeurIPS 2015. LSTM over interaction sequences predicts P(correct | skill).

Data format expected (CSV or JSON list):
──────────────────────────────────────────
  user_id, skill_tag, exercise_type, correctness, time_taken_s,
  hint_used, attention_score, error_type, time_since_review_s,
  session_position, total_questions

Usage
──────
  python ml_training/train_lstm.py \
      --data interactions.csv \
      --out models/lstm_mastery.pt \
      --epochs 30 \
      --batch 32 \
      --seq_len 20

The trained model is loaded by ml/lstm_mastery.py → LSTMMasteryPredictor.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[train_lstm] PyTorch not installed. Run: pip install torch")

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.lstm_mastery import (
    LSTMMasteryModel, SKILL_IDS, EXERCISE_IDS, ERROR_TYPES,
    N_SKILLS, N_EX_TYPES, N_ERRORS, EMBED_DIM, NUMERIC_FEATS,
)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """
    Groups interactions by user, builds fixed-length subsequences.
    Each sample = (skill_ids, extype_ids, error_ids, numeric, targets)
    all of shape (seq_len, ...).
    Target = correctness[t+1] (predict next answer from history up to t).
    """

    def __init__(self, records: List[Dict], seq_len: int = 20, augment: bool = True):
        self.seq_len = seq_len
        self.augment = augment
        self.sequences = self._build_sequences(records)

    def _build_sequences(self, records: List[Dict]) -> List[Tuple]:
        # Group by user
        user_records: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            user_records[r["user_id"]].append(r)

        seqs = []
        for uid, recs in user_records.items():
            # Sort by timestamp if available
            recs.sort(key=lambda r: r.get("ts", 0))
            # Slide window over the user's full history
            for start in range(0, max(1, len(recs) - 1), self.seq_len // 2):
                chunk = recs[start: start + self.seq_len + 1]
                if len(chunk) < 3:
                    continue
                # Inputs = all but last, Target = correctness of [1:]
                inputs = chunk[:-1]
                targets_raw = [r["correctness"] for r in chunk[1:]]
                # Pad to seq_len
                inputs  = self._pad(inputs,  self.seq_len)
                targets = self._pad_floats(targets_raw, self.seq_len)
                seqs.append(self._encode(inputs, targets))

        return seqs

    def _pad(self, recs: List[Dict], length: int) -> List[Dict]:
        blank = {
            "skill_tag": "vocabulary_basic",
            "exercise_type": "multiple_choice_vocab",
            "error_type": "none",
            "correctness": 0.0, "time_taken_s": 10.0, "hint_used": 0,
            "attention_score": 0.5, "time_since_review_s": 0.0,
            "session_position": 1, "total_questions": 10,
        }
        while len(recs) < length:
            recs = [blank] + recs
        return recs[-length:]

    def _pad_floats(self, vals: List[float], length: int) -> List[float]:
        while len(vals) < length:
            vals = [0.0] + vals
        return vals[-length:]

    def _encode(self, recs: List[Dict], targets: List[float]) -> Tuple:
        skill_ids  = [SKILL_IDS.get(r["skill_tag"], 0)    for r in recs]
        extype_ids = [EXERCISE_IDS.get(r["exercise_type"], 0) for r in recs]
        error_ids  = [ERROR_TYPES.get(r.get("error_type", "none"), 0) for r in recs]
        numeric    = []
        for i, r in enumerate(recs):
            pos = r.get("session_position", i + 1)
            tot = max(1, r.get("total_questions", 10))
            numeric.append([
                float(r.get("correctness", 0)),
                min(1.0, float(r.get("time_taken_s", 10)) / 60.0),
                float(r.get("hint_used", 0)),
                float(r.get("attention_score", 0.5)),
                min(1.0, float(r.get("time_since_review_s", 0)) / (7 * 86400)),
                (pos - 1) / (tot - 1) if tot > 1 else 0.0,
            ])
        return (
            np.array(skill_ids,  dtype=np.int64),
            np.array(extype_ids, dtype=np.int64),
            np.array(error_ids,  dtype=np.int64),
            np.array(numeric,    dtype=np.float32),
            np.array(targets,    dtype=np.float32),
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sk, ex, er, num, tgt = self.sequences[idx]
        return (
            torch.from_numpy(sk),
            torch.from_numpy(ex),
            torch.from_numpy(er),
            torch.from_numpy(num),
            torch.from_numpy(tgt),
        )


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_records(path: str) -> List[Dict]:
    path = Path(path)
    if path.suffix == ".json":
        with open(path) as f:
            records = json.load(f)
    else:  # CSV
        records = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "user_id":            row.get("user_id", "u0"),
                    "skill_tag":          row.get("skill_tag", "vocabulary_basic"),
                    "exercise_type":      row.get("exercise_type", "multiple_choice_vocab"),
                    "correctness":        float(row.get("correctness", 0)),
                    "time_taken_s":       float(row.get("time_taken_s", 10)),
                    "hint_used":          int(row.get("hint_used", 0)),
                    "attention_score":    float(row.get("attention_score", 0.5)),
                    "error_type":         row.get("error_type", "none"),
                    "time_since_review_s": float(row.get("time_since_review_s", 0)),
                    "session_position":   int(row.get("session_position", 1)),
                    "total_questions":    int(row.get("total_questions", 10)),
                    "ts":                 float(row.get("ts", time.time())),
                })
    print(f"[Data] Loaded {len(records)} interactions.")
    return records


def generate_synthetic_data(n_users: int = 100, n_questions: int = 50) -> List[Dict]:
    """
    Generate synthetic interaction data for initial training / testing.

    Simulates ADHD-like learning patterns:
    - Accuracy degrades over long sessions (fatigue)
    - Attention score correlates with accuracy
    - Some skills are harder (grammar > vocabulary)
    - Spaced practice improves retention

    This data can be used to pre-train the LSTM before real user data
    is collected. A common technique in EdTech (CognitiveTutor, ASSISTments).
    """
    records = []
    skills = list(SKILL_IDS.keys())
    ex_types = list(EXERCISE_IDS.keys())
    error_types = list(ERROR_TYPES.keys())

    # Skill difficulty priors (0=easy, 1=hard)
    skill_difficulty = {
        "vocabulary_basic": 0.1, "vocabulary_intermediate": 0.3,
        "grammar_present": 0.4, "grammar_past": 0.5, "grammar_future": 0.5,
        "grammar_genders": 0.6, "grammar_plurals": 0.4, "grammar_adjectives": 0.5,
        "grammar_pronouns": 0.55, "reading_comprehension": 0.45,
        "translation_basic": 0.3, "translation_intermediate": 0.6,
        "listening_comprehension": 0.5, "sentence_structure": 0.55,
        "numbers_time": 0.2, "social_phrases": 0.15,
    }

    for uid in range(n_users):
        # Each user has a random mastery profile
        user_mastery = {s: max(0.1, min(0.9, random.gauss(0.4, 0.2)))
                        for s in skills}
        last_seen = {s: 0.0 for s in skills}

        session_start = time.time() - random.randint(0, 7 * 86400)
        for q in range(n_questions):
            skill = random.choices(skills, weights=[1.0]*len(skills))[0]
            ex_type = random.choice(ex_types)

            # Fatigue: accuracy drops after Q15
            fatigue = max(0, (q - 15) / n_questions)

            # Attention (ADHD: high variance, correlated with fatigue)
            attention = max(0.1, min(1.0, random.gauss(0.65 - fatigue * 0.3, 0.2)))

            # Mastery update (simplified learning curve)
            base_mastery = user_mastery[skill]
            diff = skill_difficulty.get(skill, 0.4)
            p_correct = min(0.95, max(0.05,
                base_mastery * (1 - diff * 0.5) * attention * (1 - fatigue * 0.4)
            ))
            correct = random.random() < p_correct

            # Update mastery slightly
            if correct:
                user_mastery[skill] = min(0.99, user_mastery[skill] + 0.02)
            else:
                user_mastery[skill] = max(0.01, user_mastery[skill] - 0.01)

            # Spaced review effect
            time_since = session_start - last_seen[skill]
            # Forgetting: mastery decays slowly without review
            if time_since > 3 * 86400:
                user_mastery[skill] = max(0.1, user_mastery[skill] - 0.05)

            rt = max(1.0, random.gauss(12 - base_mastery * 8, 5))
            error_type = "none" if correct else random.choice(
                ["vocabulary", "grammar", "word_order", "spelling"]
            )

            records.append({
                "user_id":            f"user_{uid:04d}",
                "skill_tag":          skill,
                "exercise_type":      ex_type,
                "correctness":        1.0 if correct else 0.0,
                "time_taken_s":       round(rt, 2),
                "hint_used":          1 if random.random() < 0.15 else 0,
                "attention_score":    round(attention, 3),
                "error_type":         error_type,
                "time_since_review_s": round(time_since, 1),
                "session_position":   q + 1,
                "total_questions":    n_questions,
                "ts":                 session_start + q * 45,
            })
            last_seen[skill] = session_start + q * 45

    print(f"[Synthetic] Generated {len(records)} records for {n_users} users.")
    return records


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(
    data_path:  Optional[str],
    output_path: str = "models/lstm_mastery.pt",
    epochs:     int = 30,
    batch_size: int = 32,
    seq_len:    int = 20,
    lr:         float = 1e-3,
    val_split:  float = 0.15,
    synthetic_users: int = 200,
):
    if not TORCH_OK:
        print("PyTorch required. pip install torch")
        return

    # Load data
    if data_path and Path(data_path).exists():
        records = load_records(data_path)
    else:
        print("[Data] No data file found — generating synthetic training data.")
        print("       Research: Piech et al. 2015 — synthetic pretraining is")
        print("       standard practice before real user data is available.")
        records = generate_synthetic_data(n_users=synthetic_users, n_questions=60)

    # Split
    users = list({r["user_id"] for r in records})
    random.shuffle(users)
    val_users = set(users[:int(len(users) * val_split)])
    train_records = [r for r in records if r["user_id"] not in val_users]
    val_records   = [r for r in records if r["user_id"] in val_users]

    train_ds = InteractionDataset(train_records, seq_len=seq_len)
    val_ds   = InteractionDataset(val_records,   seq_len=seq_len, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[Data] Train: {len(train_ds)} seqs | Val: {len(val_ds)} seqs")

    # Model
    model = LSTMMasteryModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss, train_auc = 0.0, []
        for sk, ex, er, num, tgt in train_dl:
            optimizer.zero_grad()
            pred, _ = model(sk, ex, er, num)
            pred = pred.squeeze(-1)   # (B, T)
            loss = criterion(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Val ────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for sk, ex, er, num, tgt in val_dl:
                pred, _ = model(sk, ex, er, num)
                pred = pred.squeeze(-1)
                val_loss += criterion(pred, tgt).item()
                all_preds.extend(pred.flatten().tolist())
                all_tgts.extend(tgt.flatten().tolist())

        avg_train = train_loss / max(1, len(train_dl))
        avg_val   = val_loss   / max(1, len(val_dl))

        # AUC approximation (rank correlation)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_tgts, all_preds)
        except Exception:
            auc = 0.0

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | AUC={auc:.3f}")

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ Saved best model → {output_path}")

    print(f"\n[Done] Best val loss: {best_val_loss:.4f}")
    print(f"[Done] Model saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Collect real user interaction data via /api/lang/progress/update")
    print("  2. Export interactions to CSV and re-run this script")
    print("  3. Load trained model: LSTMMasteryPredictor('models/lstm_mastery.pt')")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuroLearn LSTM Mastery Model")
    parser.add_argument("--data",    type=str, default=None,
                        help="Path to interactions CSV/JSON (optional; uses synthetic if absent)")
    parser.add_argument("--out",     type=str, default="models/lstm_mastery.pt")
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--synthetic_users", type=int, default=200,
                        help="Number of synthetic users if no real data")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        seq_len=args.seq_len,
        lr=args.lr,
        synthetic_users=args.synthetic_users,
    )
