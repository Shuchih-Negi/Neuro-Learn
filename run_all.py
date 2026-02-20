"""
run_all.py
==========
One-shot script for hackathon demos.

  python run_all.py --mode train    → generate data, train LSTM, save model
  python run_all.py --mode demo     → load saved model, run live inference demo
  python run_all.py --mode full     → train + demo (default)
  python run_all.py --mode rules    → skip LSTM, show rule-based only

Usage with real ASSISTments data:
  python run_all.py --mode full --data path/to/skill_builder_data_corrected.csv
"""

import argparse
import sys
import os

# Add project root to path so 'utils' and 'models' are importable
# Works on both Windows and Unix regardless of where you run the script from
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader         import generate_synthetic_dataset, load_assistments
from utils.feature_engineering import build_features, FEATURE_COLS, INT_TO_STATE


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────

def run_train(data_path=None):
    if data_path:
        print(f"Loading real ASSISTments data from {data_path}…")
        raw = load_assistments(data_path)
        label_col = "label_auto"     # rule-based labels for real data
    else:
        print("Generating synthetic dataset (300 students)…")
        raw = generate_synthetic_dataset(n_students=300)
        label_col = "label_true"     # ground-truth labels for synthetic data

    print("\nBuilding feature matrix…")
    feat = build_features(raw, label_col=label_col)

    print("\nTraining LSTM model…")
    from models.lstm_model import train_model
    model, scaler = train_model(feat, save_dir="models/")
    print("\n✅  Training complete. Model saved to models/")
    return feat


# ─────────────────────────────────────────────────────────────
# DEMO INFERENCE
# ─────────────────────────────────────────────────────────────

def run_demo(feat=None, use_lstm=True):
    print("\n" + "="*60)
    print("  LIVE INFERENCE DEMO  (simulating real-time API calls)")
    print("="*60)

    if feat is None:
        raw  = generate_synthetic_dataset(n_students=10, seed=99)
        feat = build_features(raw, label_col="label_true")

    if use_lstm:
        try:
            from models.lstm_model import AttentionPredictor
            predictor = AttentionPredictor("models/best_lstm.pt", "models/scaler.pkl")
            mode = "LSTM"
        except Exception as e:
            print(f"LSTM load failed ({e}), falling back to rule-based")
            predictor = None
            mode = "rule-based"
    else:
        predictor = None
        mode = "rule-based"

    from utils.feature_engineering import rule_based_state

    student_id = feat["user_id"].unique()[0]
    rows       = feat[feat["user_id"] == student_id].sort_values("order_id")

    print(f"\nStudent {student_id} | Mode: {mode}\n")
    print(f"{'Q#':>4}  {'True State':15} {'Pred State':15} {'Conf':>6}  {'Action':25}  {'Stability':>9}")
    print("-" * 80)

    for _, row in rows.iterrows():
        if predictor:
            result = predictor.push(student_id, row[FEATURE_COLS].to_dict())
            if result["state"] is None:
                continue
            pred_state  = result["state"]
            confidence  = result["confidence"]
            action      = result["action"]
        else:
            # Pure rule-based
            pred_state = rule_based_state(
                rt_s          = row["rt_s"],
                correct       = row["correct"],
                attempt_count = row["attempt_count"],
                hint_count    = row["hint_count"],
                idle_s        = row["idle_s"],
                rt_variance   = row["rt_variance"],
                error_burst   = row["error_burst"],
            )
            confidence = 0.90
            action_map = {
                "Focused":     "increase_difficulty",
                "Drifting":    "shorter_task",
                "Impulsive":   "add_scaffold",
                "Overwhelmed": "simplify_problem",
            }
            action = action_map[pred_state]

        true_state = row.get("label_true", "?")
        stability  = row["stability_score"]
        match      = "✓" if pred_state == true_state else "✗"

        print(f"{row['order_id']:>4}  "
              f"{str(true_state):15} "
              f"{pred_state:15} "
              f"{confidence:>5.0%}  "
              f"{action:25}  "
              f"{stability:>7.1f}  {match}")


# ─────────────────────────────────────────────────────────────
# RULES ONLY (hackathon minimum viable path)
# ─────────────────────────────────────────────────────────────

def run_rules_demo():
    print("\n=== Rule-Based Classifier Demo ===\n")

    test_cases = [
        {"rt_s": 2.1,  "correct": 0, "attempt_count": 1, "hint_count": 0, "idle_s": 0.5, "rt_variance": 5,  "error_burst": 0.0, "expected": "Impulsive"},
        {"rt_s": 28.0, "correct": 0, "attempt_count": 2, "hint_count": 1, "idle_s": 15,  "rt_variance": 80, "error_burst": 0.5, "expected": "Drifting"},
        {"rt_s": 40.0, "correct": 0, "attempt_count": 4, "hint_count": 3, "idle_s": 10,  "rt_variance": 30, "error_burst": 0.7, "expected": "Overwhelmed"},
        {"rt_s": 7.5,  "correct": 1, "attempt_count": 1, "hint_count": 0, "idle_s": 1.0, "rt_variance": 10, "error_burst": 0.0, "expected": "Focused"},
    ]

    from utils.feature_engineering import rule_based_state

    print(f"{'RT(s)':>7}  {'Correct':>7}  {'Attempts':>8}  {'Hints':>5}  {'Expected':15} {'Predicted':15} {'Match'}")
    print("-" * 75)

    for case in test_cases:
        pred = rule_based_state(
            rt_s          = case["rt_s"],
            correct       = case["correct"],
            attempt_count = case["attempt_count"],
            hint_count    = case["hint_count"],
            idle_s        = case["idle_s"],
            rt_variance   = case["rt_variance"],
            error_burst   = case["error_burst"],
        )
        match = "✓" if pred == case["expected"] else "✗"
        print(f"{case['rt_s']:>7.1f}  "
              f"{case['correct']:>7}  "
              f"{case['attempt_count']:>8}  "
              f"{case['hint_count']:>5}  "
              f"{case['expected']:15} "
              f"{pred:15} {match}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  default="full",
                        choices=["train","demo","full","rules"],
                        help="What to run")
    parser.add_argument("--data",  default=None,
                        help="Path to ASSISTments CSV (optional)")
    args = parser.parse_args()

    if args.mode == "rules":
        run_rules_demo()

    elif args.mode == "train":
        run_train(args.data)

    elif args.mode == "demo":
        run_demo(use_lstm=True)

    elif args.mode == "full":
        feat = run_train(args.data)
        run_demo(feat=feat, use_lstm=True)
