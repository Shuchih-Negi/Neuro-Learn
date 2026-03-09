# NeuroLearn — Language Edition
## Neuro-Adaptive Gamified Language Learning for ADHD Students

---

## What Was Added (Language Expansion)

NeuroLearn started as an adaptive math tutor. This expansion adds a full language learning pipeline while keeping every existing math endpoint intact.

```
Original:   /api/next_question  /api/story  /api/feedback  /api/progress  /api/dashboard
New:        /api/lang/*  (parallel namespace — no breaking changes)
```

---

## Research Foundation

### 1. Language Acquisition
| Theory | Author | Applied In |
|--------|--------|-----------|
| Comprehensible Input (i+1) | Krashen (1982) | ReasoningAgent difficulty selection |
| Output Hypothesis | Swain (1985) | Every exercise requires active production |
| Recast Feedback | Lyster & Ranta (1997) | QA + Feedback agents — never "wrong", always recast |
| Spaced Repetition | Ebbinghaus (1885) | SM-2 algorithm in `/api/lang/progress/update` |
| Varied Context | Nation (2001) | Exercise types rotate even for same vocabulary |

### 2. ADHD & Cognitive Load
| Finding | Author | Applied In |
|---------|--------|-----------|
| Executive function & WM limits | Toplak et al. (2013) | 25-word max instructions, micro-sessions |
| Working Memory Model | Baddeley & Hitch (1974) | Overwhelmed state → single-step exercises |
| Multimedia Learning | Mayer (2009) | `visual_breakdown` field (emoji grammar patterns) |
| Dopamine & ADHD reward | Lam & Muldner (2018) | `dopamine_reward` delivered instantly on correct answer |

### 3. Knowledge Tracing
| Model | Author | Applied In |
|-------|--------|-----------|
| Bayesian KT | Corbett & Anderson (1994) | EWMA mastery fallback (pre-LSTM) |
| Deep Knowledge Tracing | Piech et al. (2015) | `ml/lstm_mastery.py` LSTM architecture |
| Forgetting Curve | Ebbinghaus (1885) | `time_since_review_s` LSTM feature |
| SM-2 Spaced Repetition | Wozniak (1987) | `_sm2_next_review()` in app.py |

---

## File Structure

```
NeuroLearn/
│
├── agents/
│   └── moderator.py          ← EXPANDED: full language multi-agent pipeline
│                               Agents: Reasoning / LangQuestion / Story / QA / Hint
│
├── backend/
│   └── app.py                ← EXPANDED: all original math endpoints + /api/lang/*
│
├── ml/
│   ├── attention_model.py    ← UNCHANGED (your original)
│   ├── eye_tracker.py        ← UNCHANGED (your original)
│   ├── lstm_mastery.py       ← NEW: LSTM knowledge tracing for language skills
│   └── nlp_model.py          ← NEW: grammar correction + semantic answer validation
│
├── ml_training/
│   └── train_lstm.py         ← NEW: full training pipeline (synthetic + real data)
│
├── models/
│   └── lstm_model.pt         ← your existing LSTM attention model
│
├── requirements.txt
└── README.md
```

---

## API Reference

### Language Endpoints

#### `GET /api/lang/languages`
Returns all supported languages, exercise types, and skill tags.

#### `POST /api/lang/next_question`
Main endpoint. Full 5-agent pipeline.

```json
{
  "character": "Luna",
  "topic": "Spanish greetings and farewells",
  "target_language": "es",
  "difficulty": 1,
  "attention_state": "Focused",
  "attention_confidence": 0.82,
  "skill_focus": "vocabulary_basic",
  "learner_age": 12,
  "student_id": "student_001",
  "question_number": 1,
  "total_questions": 8,
  "session_accuracy": 0.0,
  "session_fatigue": 0.0,
  "eye_metrics": {
    "blink_rate": 18.0,
    "fixation_duration": 340.0,
    "gaze_stability": 0.85
  }
}
```

Response includes:
- `question`, `options`, `correct_index`, `explanation`, `hints`
- `skill_tag` — post to progress tracker
- `grammar_explanation` — ≤30 word rule
- `visual_breakdown` — emoji pattern
- `dopamine_reward` — instant feedback message
- `acceptable_answers` — list of valid semantic variants
- `mastery_hint` — spaced rep recommendation

#### `POST /api/lang/story/generate`
Immersion story in the target language.

```json
{
  "character": "Luna",
  "topic": "ordering food at a café",
  "target_language": "fr",
  "difficulty": 2,
  "learner_age": 12
}
```

#### `POST /api/lang/progress/update`
Post after every answer. Updates mastery + spaced-rep schedule.

```json
{
  "student_id": "student_001",
  "skill_tag": "vocabulary_basic",
  "correct": true,
  "response_time": 7.3,
  "attention_state": "Focused",
  "xp_earned": 25,
  "exercise_type": "multiple_choice_vocab",
  "target_language": "es",
  "section_id": "session_001"
}
```

Response:
```json
{
  "status": "ok",
  "skill_mastery": 0.62,
  "next_review_in_days": 3,
  "mastery_scores": { "vocabulary_basic": 0.62, "grammar_present": 0.41 }
}
```

Feed `mastery_scores` back into the next `/api/lang/next_question` call.

#### `POST /api/lang/feedback/generate`
End-of-session feedback.

#### `GET /api/lang/mastery/{student_id}`
Full mastery state + SM-2 review schedule for all skills.

#### `GET /api/lang/dashboard/{student_id}`
Full dashboard: accuracy, streaks, mastery radar, attention distribution, ADHD insights.

---

## Adaptive Logic

The `ReasoningAgent` applies these rules (from the blueprint + Krashen):

| Condition | Action |
|-----------|--------|
| mastery < 0.35 | `reinforce` — repeat same skill, easier type |
| mastery 0.35–0.65 | `extend` — varied contexts for same skill |
| mastery > 0.65 | `advance` — move to harder skill or combine |
| attention = Overwhelmed | difficulty −1, single-step, ≤12 word question |
| attention = Drifting | same difficulty + novelty/surprise hook |
| attention = Impulsive | add deliberate trap option + "Look carefully" |
| attention = Focused | multi-step, production-oriented exercise |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key
export GEMINI_API_KEY=your_key_here

# 3. Run the backend
uvicorn backend.app:app --reload --port 8000

# 4. (Optional) Train LSTM mastery model
python ml_training/train_lstm.py --out models/lstm_mastery.pt

# 5. (Optional) Install NLP semantic validator
pip install sentence-transformers
```

---

## Supported Languages
Spanish · French · German · Italian · Portuguese · Japanese · Mandarin · Arabic · Hindi · Korean

---

## Supported Exercise Types
- `multiple_choice_vocab` — translate / define
- `fill_in_the_blank` — complete the sentence
- `translation` — produce a short phrase
- `grammar_sort` — correct conjugation / gender / plural
- `listening_text` — comprehension MCQ
- `match_pairs` — word ↔ meaning
- `sentence_builder` — correct word order

---

*Built for hackathons, EdTech research, and cognitive learning innovation.*
