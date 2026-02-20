# Adaptive Attention ML — Hackathon Edition

## Project Structure

```
attention_ml/
├── utils/
│   ├── data_loader.py          # ASSISTments loader + synthetic data generator
│   └── feature_engineering.py  # Rolling-window features + rule-based labeler
├── models/
│   └── lstm_model.py           # PyTorch LSTM + live AttentionPredictor class
├── api.py                      # FastAPI backend (connects to React frontend)
├── run_all.py                  # One-shot train + demo script
└── README.md
```

---

## Dataset Options (pick one)

### Option A — Real ASSISTments data (recommended for judges)
1. Download: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
2. File: `skill_builder_data_corrected.csv`
3. Run: `python run_all.py --mode full --data skill_builder_data_corrected.csv`

**What it contains:**
- 525,000+ interactions from 4,217 students
- Columns: `user_id`, `correct`, `attempt_count`, `ms_first_response`, `hint_count`
- Math problems grades 5–12

### Option B — Synthetic data (works offline, zero setup)
```bash
python run_all.py --mode full
```
Generates 300 students with embedded attention state patterns.

---

## Quickstart

```bash
pip install torch pandas numpy scikit-learn fastapi uvicorn pydantic

# Option 1: Just see the rule-based classifier working (30 seconds)
python run_all.py --mode rules

# Option 2: Full train + demo on synthetic data (~2-3 mins)
python run_all.py --mode full

# Option 3: Start the API server
uvicorn api:app --reload --port 8000
```

---

## API Reference

### POST /log-interaction
Receives a question event, returns attention prediction.

```json
{
  "user_id": 42,
  "question_id": 1001,
  "correct": 0,
  "response_time_s": 2.3,
  "attempt_count": 1,
  "hint_count": 0,
  "overlap_time": 3.1
}
```

Response:
```json
{
  "state": "Impulsive",
  "confidence": 0.82,
  "action": "add_scaffold",
  "stability_score": 61.4,
  "probabilities": {
    "Focused": 0.05,
    "Drifting": 0.07,
    "Impulsive": 0.82,
    "Overwhelmed": 0.06
  }
}
```

### GET /attention-state/{user_id}
Returns the latest state for a student.

### GET /stability-history/{user_id}
Returns time-series for the attention graph widget.

### GET /parent-dashboard/{user_id}
Weekly analytics summary.

### GET /teacher-heatmap/{class_id}
Class-wide aggregation with intervention alerts.

---

## ML Architecture

```
Phase 1 (Rule-Based) — runs immediately, no training needed:
  rule_based_state(rt_s, correct, attempts, hints, idle, rt_variance, error_burst)
  → one of: Focused | Drifting | Impulsive | Overwhelmed

Phase 2 (LSTM) — train first, then swap in automatically:
  Input:  [batch, seq_len=10, features=13]
  LSTM(64 units, 2 layers) → Dropout(0.3)
  Dense(32, ReLU) → Dense(4, Softmax)
  Output: P(Focused), P(Drifting), P(Impulsive), P(Overwhelmed)
```

## Attention States → UI Actions

| State       | Trigger                        | UI Action             |
|-------------|--------------------------------|-----------------------|
| Focused     | Fast RT + high accuracy        | Increase difficulty   |
| Drifting    | Slow RT + idle spikes          | Shorter task          |
| Impulsive   | Very fast RT + wrong, no hints | Add scaffold / hint   |
| Overwhelmed | Many retries + many hints      | Simplify problem      |

## Stability Score Formula
```
score = 100 − (0.30 × rt_variance_norm + 0.40 × error_burst + 0.30 × idle_norm)
```
Rendered as a 0–100 animated line chart in the student view.
