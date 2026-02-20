"""
lstm_model.py
=============
PyTorch LSTM that classifies student attention state from a
sequence of interaction features.

Architecture (from the spec):
  Input:  [batch, seq_len, n_features]
  LSTM(64) → Dropout(0.3) → Dense(32) → Dense(4, softmax)
  Output: P(Focused), P(Drifting), P(Impulsive), P(Overwhelmed)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path

from utils.feature_engineering import FEATURE_COLS, STATE_TO_INT, INT_TO_STATE


# ─────────────────────────────────────────────────────────────
# HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────────

SEQ_LEN    = 10      # how many past questions feed each prediction
INPUT_DIM  = len(FEATURE_COLS)
HIDDEN_DIM = 64
N_CLASSES  = 4
DROPOUT    = 0.3
BATCH_SIZE = 128
EPOCHS     = 30
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class AttentionSequenceDataset(Dataset):
    """
    Builds fixed-length sliding-window sequences per student.
    Each sample is (seq_features [SEQ_LEN, INPUT_DIM], label [int])
    where label is the state at the LAST step of the window.
    """

    def __init__(self, feat_df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = False):
        sequences, labels = [], []

        # Normalise features
        X_raw = feat_df[FEATURE_COLS].values.astype(np.float32)
        if fit_scaler:
            self.scaler = StandardScaler()
            X_raw = self.scaler.fit_transform(X_raw)
        elif scaler is not None:
            self.scaler = scaler
            X_raw = scaler.transform(X_raw)
        else:
            self.scaler = None

        feat_df = feat_df.copy()
        feat_df[FEATURE_COLS] = X_raw

        # Build sequences per student
        for uid, group in feat_df.groupby("user_id"):
            group = group.sort_values("order_id").reset_index(drop=True)
            x_mat = group[FEATURE_COLS].values.astype(np.float32)
            y_vec = group["y"].values.astype(np.int64)

            for i in range(SEQ_LEN, len(group)):
                seq = x_mat[i - SEQ_LEN : i]    # shape (SEQ_LEN, INPUT_DIM)
                lbl = y_vec[i]
                sequences.append(seq)
                labels.append(lbl)

        self.X = torch.tensor(np.array(sequences), dtype=torch.float32)
        self.y = torch.tensor(np.array(labels),    dtype=torch.long)
        print(f"  Dataset: {len(self.X):,} sequences | shape {tuple(self.X.shape)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

class AttentionLSTM(nn.Module):
    def __init__(
        self,
        input_dim:  int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_classes:  int = N_CLASSES,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            dropout     = dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_dim, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden_dim)
        out    = out[:, -1, :]         # take last timestep
        out    = self.dropout(out)
        out    = self.relu(self.fc1(out))
        out    = self.fc2(out)         # raw logits
        return out                     # CrossEntropyLoss expects logits


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train_model(feat_df: pd.DataFrame, save_dir: str = "models/"):
    """
    Full training pipeline:
      1. Split by student (no leakage)
      2. Build datasets + dataloaders
      3. Train LSTM
      4. Evaluate on held-out students
      5. Save model + scaler

    Returns trained model and scaler.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nTraining on device: {DEVICE}")

    # ── Student-level train/val split ────────────────────────
    all_users  = feat_df["user_id"].unique()
    train_uids, val_uids = train_test_split(all_users, test_size=0.2, random_state=42)

    train_df = feat_df[feat_df["user_id"].isin(train_uids)]
    val_df   = feat_df[feat_df["user_id"].isin(val_uids)]

    print(f"Train students: {len(train_uids)} | Val students: {len(val_uids)}")

    # ── Datasets ─────────────────────────────────────────────
    print("Building train dataset…")
    train_ds = AttentionSequenceDataset(train_df, fit_scaler=True)
    print("Building val dataset…")
    val_ds   = AttentionSequenceDataset(val_df, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model, loss, optimiser ───────────────────────────────
    model     = AttentionLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_acc = 0.0

    # ── Training ─────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item() * len(y_batch)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += len(y_batch)

        # ── Validation ───────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_true   = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                preds  = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total   += len(y_batch)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        avg_loss  = train_loss    / train_total
        scheduler.step(1 - val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"loss={avg_loss:.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/best_lstm.pt")
            with open(f"{save_dir}/scaler.pkl", "wb") as f:
                pickle.dump(train_ds.scaler, f)

    # ── Final classification report ──────────────────────────
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    target_names = [INT_TO_STATE[i] for i in range(N_CLASSES)]
    print("\nClassification Report (best model on val set):")
    print(classification_report(all_true, all_preds, target_names=target_names))

    # Reload best weights before returning
    model.load_state_dict(torch.load(f"{save_dir}/best_lstm.pt", map_location=DEVICE))
    return model, train_ds.scaler


# ─────────────────────────────────────────────────────────────
# INFERENCE  (used by the live API endpoint)
# ─────────────────────────────────────────────────────────────

class AttentionPredictor:
    """
    Stateful predictor that wraps a trained LSTM.
    Maintains a rolling feature buffer per student for real-time use.
    """

    def __init__(self, model_path: str, scaler_path: str):
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.model = AttentionLSTM().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        # Per-student rolling buffer  {user_id → list of feature vectors}
        self._buffers: dict = {}

    def push(self, user_id: int, feature_dict: dict) -> dict:
        """
        Receive one new interaction log entry for a student.
        Returns the predicted attention state (or None if not enough history).

        feature_dict keys must match FEATURE_COLS.
        """
        vec = np.array([feature_dict.get(c, 0.0) for c in FEATURE_COLS], dtype=np.float32)
        vec_scaled = self.scaler.transform(vec.reshape(1, -1))[0]

        buf = self._buffers.setdefault(user_id, [])
        buf.append(vec_scaled)

        if len(buf) < SEQ_LEN:
            return {"state": None, "confidence": None, "probabilities": None}

        # Trim buffer to SEQ_LEN
        self._buffers[user_id] = buf[-SEQ_LEN:]

        seq   = torch.tensor(np.array(buf[-SEQ_LEN:]), dtype=torch.float32)
        seq   = seq.unsqueeze(0).to(DEVICE)           # (1, SEQ_LEN, INPUT_DIM)

        with torch.no_grad():
            logits = self.model(seq)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx    = int(np.argmax(probs))
        pred_state  = INT_TO_STATE[pred_idx]
        confidence  = float(probs[pred_idx])

        return {
            "state":         pred_state,
            "confidence":    round(confidence, 3),
            "probabilities": {INT_TO_STATE[i]: round(float(probs[i]), 3) for i in range(N_CLASSES)},
            "action":        _state_to_action(pred_state),
        }


def _state_to_action(state: str) -> str:
    return {
        "Focused":     "increase_difficulty",
        "Drifting":    "shorter_task",
        "Impulsive":   "add_scaffold",
        "Overwhelmed": "simplify_problem",
    }.get(state, "no_action")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from utils.data_loader       import generate_synthetic_dataset
    from utils.feature_engineering import build_features

    print("=== Generating synthetic data ===")
    raw  = generate_synthetic_dataset(n_students=100)

    print("\n=== Building features ===")
    feat = build_features(raw, label_col="label_true")

    print("\n=== Training LSTM ===")
    model, scaler = train_model(feat, save_dir="models/")

    print("\n=== Live inference demo ===")
    predictor = AttentionPredictor("models/best_lstm.pt", "models/scaler.pkl")

    sample_student = 0
    student_rows   = feat[feat["user_id"] == sample_student].sort_values("order_id")

    for _, row in student_rows.head(15).iterrows():
        result = predictor.push(
            user_id      = sample_student,
            feature_dict = row[FEATURE_COLS].to_dict(),
        )
        if result["state"]:
            true_lbl = row.get("label_true", "?")
            print(f"  Q{row['order_id']:4d} | True: {true_lbl:12s} | "
                  f"Pred: {result['state']:12s} | "
                  f"Conf: {result['confidence']:.2f} | "
                  f"Action: {result['action']}")
