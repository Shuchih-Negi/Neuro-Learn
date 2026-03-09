"""
backend/app.py  — NeuroLearn (Math + Language Edition)
=======================================================
FastAPI server.  All original math endpoints preserved.
New /api/lang/* namespace added for language learning.

Run:  uvicorn backend.app:app --reload --port 8000
Env:  GEMINI_API_KEY

Architecture
────────────
  /api/health                      — health check
  /api/next_question               — original math adaptive question (unchanged)
  /api/story/generate              — original math story (unchanged)
  /api/questions/generate          — original math question (unchanged)
  /api/feedback/generate           — original math feedback (unchanged)
  /api/evaluate/generate           — original math evaluation (unchanged)
  /api/progress/update             — original progress tracking (unchanged)
  /api/dashboard/{student_id}      — original dashboard (unchanged)

  ── Language Learning (NEW) ────────────────────────────────────
  /api/lang/languages              — list supported languages
  /api/lang/next_question          — adaptive language question (full pipeline)
  /api/lang/story/generate         — language immersion story
  /api/lang/feedback/generate      — language session feedback
  /api/lang/progress/update        — language progress + mastery tracking
  /api/lang/dashboard/{student_id} — language learning dashboard
  /api/lang/mastery/{student_id}   — per-skill mastery scores for LSTM input

Research integration
────────────────────
  • Progress endpoint updates mastery_scores dict which feeds back into the
    ReasoningAgent's Krashen i+1 difficulty decisions next question.
  • Spaced-repetition next_review timestamps computed via SM-2 algorithm.
  • All session data stored in memory (swap for Redis/Postgres in production).
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Enhanced Gemini integration
from gemini_utils import (
    get_gemini_client,
    call_gemini_json,
    call_gemini_text,
    create_story_model,
    create_question_model,
    create_feedback_model,
    create_reasoning_model,
    GeminiConfig
)

sys.path.append("..")
from agents.moderator import (
    Moderator,
    SessionContext,
    SUPPORTED_LANGUAGES,
    EXERCISE_TYPES,
    SKILL_TAGS,
)

# Import NLP model for answer validation
sys.path.append("../ml")
from nlp_model import NLPModel

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDkfYumsK1Z02i48fpCKlxhDQLTDdVdjbM")

# Enhanced Gemini configuration
gemini_config = GeminiConfig(
    api_key=API_KEY,
    model_name=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0.7,
    max_retries=3,
    requests_per_minute=60,
)

# Initialize enhanced Gemini clients
story_client = create_story_model()
question_client = create_question_model()
feedback_client = create_feedback_model()
reasoning_client = create_reasoning_model()

# Legacy configuration for backward compatibility
genai.configure(api_key=API_KEY)

_math_moderator = None
_lang_moderator = None
_nlp_model = None


def _get_math_moderator() -> Moderator:
    global _math_moderator
    if _math_moderator is None:
        if not API_KEY:
            raise HTTPException(503, "GEMINI_API_KEY not set")
        # Import original math moderator
        from agents.moderator import Moderator as LangMod  # same class, language aware
        _math_moderator = LangMod(api_key=API_KEY, verbose=False)
    return _math_moderator


def _get_lang_moderator() -> Moderator:
    global _lang_moderator
    if _lang_moderator is None:
        if not API_KEY:
            raise HTTPException(503, "GEMINI_API_KEY not set")
        _lang_moderator = Moderator(api_key=API_KEY, verbose=False)
    return _lang_moderator


def _get_nlp_model() -> NLPModel:
    """Get or create the NLP model instance."""
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = NLPModel()
    return _nlp_model


app = FastAPI(title="NeuroLearn API — Math + Language Edition")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Gemini models (using new utility clients)
# Legacy models kept for backward compatibility
story_model    = genai.GenerativeModel("gemini-2.5-flash",
    generation_config=genai.GenerationConfig(temperature=0.9))
question_model = genai.GenerativeModel("gemini-2.5-flash",
    generation_config=genai.GenerationConfig(temperature=0.8))
feedback_model = genai.GenerativeModel("gemini-2.5-flash",
    generation_config=genai.GenerationConfig(temperature=0.85))

# ── In-memory stores ────────────────────────────────────────────────────────────
math_progress_store: Dict[str, Any]    = {}   # original math store
lang_progress_store: Dict[str, Any]    = {}   # language progress
mastery_store: Dict[str, Dict]         = {}   # user → skill → mastery data


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic models — shared
# ══════════════════════════════════════════════════════════════════════════════

class EyeMetricsPayload(BaseModel):
    blink_rate:        Optional[float] = None
    pupil_dilation:    Optional[float] = None
    fixation_duration: Optional[float] = None
    saccade_rate:      Optional[float] = None
    gaze_stability:    Optional[float] = None


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic models — original math (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class StoryRequest(BaseModel):
    character:     str
    section_title: str
    section_topic: str


class QuestionRequest(BaseModel):
    character:          str
    topic:              str
    difficulty:         int = 2
    attention_state:    str = "Focused"
    question_number:    int = 1
    total_questions:    int = 10
    session_accuracy:   float = 0.0
    previous_questions: list = []
    last_rt:            Optional[float] = None
    attention_confidence: float = Field(0.5, ge=0.0, le=1.0)
    eye_metrics:        Optional[EyeMetricsPayload] = None
    recent_states:      List[str] = Field(default_factory=list)
    last_correct:       Optional[bool] = None


class FeedbackRequest(BaseModel):
    character:        str
    total_correct:    int
    total_questions:  int
    section_title:    str
    attention_history: list = []


class EvaluateRequest(BaseModel):
    character:     str
    chapter_title: str
    topic:         str


class ProgressUpdate(BaseModel):
    student_id:     str = "default"
    section_id:     str
    correct:        bool
    response_time:  float = 0
    attention_state: str = "Focused"
    xp_earned:      int = 0


class NextQuestionRequest(BaseModel):
    character:            str
    topic:                str
    difficulty:           int = Field(2, ge=1, le=5)
    attention_state:      str = "Focused"
    attention_confidence: float = Field(0.5, ge=0.0, le=1.0)
    eye_metrics:          Optional[EyeMetricsPayload] = None
    recent_states:        List[str] = Field(default_factory=list)
    last_correct:         Optional[bool] = None
    last_rt:              Optional[float] = None
    session_accuracy:     float = Field(0.0, ge=0.0, le=1.0)
    previous_questions:   List[str] = Field(default_factory=list)
    question_number:      int = Field(1, ge=1)
    total_questions:      int = Field(10, ge=1)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic models — Language (NEW)
# ══════════════════════════════════════════════════════════════════════════════

class LangNextQuestionRequest(BaseModel):
    """
    Request for an adaptive language question.
    Includes all attention signals + language-specific context.
    """
    character:            str
    topic:                str           # e.g. "Spanish greetings and farewells"
    target_language:      str = "es"   # ISO code
    difficulty:           int = Field(1, ge=1, le=5)
    attention_state:      str = "Focused"
    attention_confidence: float = Field(0.5, ge=0.0, le=1.0)
    eye_metrics:          Optional[EyeMetricsPayload] = None
    recent_states:        List[str] = Field(default_factory=list)
    last_correct:         Optional[bool] = None
    last_rt:              Optional[float] = None
    session_accuracy:     float = Field(0.0, ge=0.0, le=1.0)
    previous_questions:   List[str] = Field(default_factory=list)
    question_number:      int = Field(1, ge=1)
    total_questions:      int = Field(10, ge=1)
    skill_focus:          str = "vocabulary_basic"
    exercise_type:        Optional[str] = None
    learner_age:          int = 12
    student_id:           str = "default"
    session_fatigue:      float = Field(0.0, ge=0.0, le=1.0)


class LangStoryRequest(BaseModel):
    """Generate an immersion story in the target language."""
    character:        str
    topic:            str
    target_language:  str = "es"
    difficulty:       int = Field(1, ge=1, le=5)
    learner_age:      int = 12


class LangFeedbackRequest(BaseModel):
    character:        str
    total_correct:    int
    total_questions:  int
    target_language:  str = "es"
    topic:            str
    attention_history: List[str] = []
    skill_results:    Dict[str, float] = {}   # skill_tag → accuracy
    student_id:       str = "default"


class LangProgressUpdate(BaseModel):
    """
    Posted after every question answer.
    Feeds the SM-2 spaced-repetition scheduler and LSTM mastery tracker.
    """
    student_id:      str = "default"
    skill_tag:       str
    correct:         bool
    response_time:   float = 0.0
    attention_state: str = "Focused"
    xp_earned:       int = 0
    exercise_type:   str = "multiple_choice_vocab"
    target_language: str = "es"
    section_id:      str = "default_section"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def call_gemini_json(model, prompt: str) -> Dict:
    """Enhanced JSON call with robust parsing and error handling."""
    try:
        # Use enhanced client for better reliability
        if hasattr(model, 'generate_content'):
            # Legacy path - use original implementation
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            if "```" in text:
                for part in text.split("```"):
                    s = part.lstrip("json").strip()
                    if s.startswith("{") or s.startswith("["):
                        text = s
                        break
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start: end + 1])
            start, end = text.find("["), text.rfind("]")
            if start >= 0 and end > start:
                return {"items": json.loads(text[start: end + 1])}
            raise ValueError("No JSON found")
        else:
            # Enhanced path - use new client
            return model.generate_json(prompt)
    except json.JSONDecodeError as e:
        raise HTTPException(502, f"Gemini returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(502, f"Gemini error: {e}")


def _sm2_next_review(
    current_ef: float,
    current_interval: int,
    quality: int,  # 0-5 (0=blackout, 5=perfect)
) -> tuple[float, int]:
    """
    SM-2 spaced repetition algorithm (Wozniak, 1987).
    Returns (new_ef, new_interval_days).

    Research: Ebbinghaus (1885) Forgetting Curve — spaced retrieval
    practice is the most robust method for vocabulary retention.
    Applied in SuperMemo, Anki, Duolingo's spaced repetition engine.
    """
    if quality < 3:
        new_interval = 1
        new_ef = max(1.3, current_ef - 0.8 + 0.28 * quality - 0.02 * quality ** 2)
    else:
        if current_interval == 0:
            new_interval = 1
        elif current_interval == 1:
            new_interval = 6
        else:
            new_interval = round(current_interval * current_ef)
        new_ef = max(1.3, current_ef + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return new_ef, new_interval


def _quality_from_answer(correct: bool, response_time: float, hint_used: bool = False) -> int:
    """
    Map answer result to SM-2 quality score (0-5).
    Research: Pimsleur (1967) — quality encoding accounts for hesitation latency.
    """
    if not correct:
        return 1 if response_time > 30 else 0
    if hint_used:
        return 3
    if response_time < 5:
        return 5
    if response_time < 15:
        return 4
    return 3


def _get_mastery_scores(student_id: str) -> Dict[str, float]:
    """Return current mastery 0-1 per skill for this student."""
    if student_id not in mastery_store:
        return {}
    skills = mastery_store[student_id]
    return {tag: data.get("mastery", 0.0) for tag, data in skills.items()}


def _update_mastery(student_id: str, skill_tag: str, correct: bool,
                    response_time: float) -> Dict:
    """
    Update mastery score using a Bayesian-inspired running average
    weighted by recency. Returns updated mastery data dict.

    Research: Corbett & Anderson (1994) — Bayesian Knowledge Tracing.
    We approximate with EWMA (exponentially weighted moving average)
    as a lightweight alternative until the LSTM model is trained.
    """
    if student_id not in mastery_store:
        mastery_store[student_id] = {}
    skills = mastery_store[student_id]
    if skill_tag not in skills:
        skills[skill_tag] = {
            "mastery":          0.0,
            "interactions":     0,
            "ef":               2.5,    # SM-2 easiness factor
            "interval_days":    0,
            "next_review_ts":   time.time(),
            "correct_streak":   0,
        }
    s = skills[skill_tag]
    alpha = 0.3  # EWMA learning rate — higher = faster adaptation

    performance = 1.0 if correct else 0.0
    if response_time < 5 and correct:
        performance = 1.0
    elif response_time > 30 and correct:
        performance = 0.7  # partial credit for slow-but-correct

    s["mastery"]      = (1 - alpha) * s["mastery"] + alpha * performance
    s["interactions"] += 1
    s["correct_streak"] = s["correct_streak"] + 1 if correct else 0

    q = _quality_from_answer(correct, response_time)
    new_ef, new_interval = _sm2_next_review(s["ef"], s["interval_days"], q)
    s["ef"]           = new_ef
    s["interval_days"] = new_interval
    s["next_review_ts"] = time.time() + new_interval * 86400

    return s


# ══════════════════════════════════════════════════════════════════════════════
# Original Math Endpoints (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "NeuroLearn API — Math + Language"}


@app.post("/api/next_question")
async def next_question(body: NextQuestionRequest):
    """Original math question endpoint."""
    from agents.moderator import SessionContext as SC
    moderator = _get_math_moderator()
    eye_dict = None
    if body.eye_metrics:
        eye_dict = {k: v for k, v in body.eye_metrics.model_dump().items() if v is not None}
        if not eye_dict:
            eye_dict = None
    ctx = SC(
        character=body.character,
        topic=body.topic,
        target_language="en",
        difficulty=body.difficulty,
        attention_state=body.attention_state,
        question_number=body.question_number,
        total_questions=body.total_questions,
        previous_questions=body.previous_questions[-5:],
        session_accuracy=body.session_accuracy,
        last_rt=body.last_rt,
        attention_confidence=body.attention_confidence,
        eye_metrics=eye_dict,
        recent_states=body.recent_states[-5:],
        last_correct=body.last_correct,
    )
    pkt = moderator.run(ctx)
    return {
        "question": pkt.question,
        "options": pkt.options,
        "correct_index": pkt.correct_index,
        "explanation": pkt.explanation,
        "difficulty": pkt.difficulty,
        "hints": pkt.hints,
        "state_used": pkt.state_used,
        "reasoning": pkt.reasoning or "",
    }


@app.post("/api/story/generate")
async def generate_story(req: StoryRequest):
    """Original math story."""
    prompt = f"""You are a fun, engaging math teacher who teaches through storytelling.
Create an adventure-style lesson about "{req.section_topic}" for a student aged 11-15.
The story MUST feature {req.character} as the main character going on a quest/mission.
Section: {req.section_title}

Rules:
- Make it exciting — {req.character} encounters a math challenge during an adventure
- Explain the concept through the story naturally
- Include 2 worked examples woven into the narrative
- Keep it to 4-5 short paragraphs
- End with a summary of 3 key points
- Use simple language appropriate for ages 11-15

Return ONLY valid JSON:
{{
  "title": "An engaging quest title",
  "story": "The full story (use \\n\\n between paragraphs)",
  "key_points": ["point1", "point2", "point3"],
  "examples": [{{"problem": "example", "solution": "step by step"}}]
}}"""
    try:
        # Use enhanced story client for better reliability
        return story_client.generate_json(prompt, expected_schema={
            "title": str,
            "story": str,
            "key_points": list,
            "examples": list
        })
    except Exception as e:
        # Fallback to legacy method
        try:
            return call_gemini_json(story_model, prompt)
        except Exception:
            return {
                "title": f"{req.character}'s Math Quest",
                "story": f"{req.character} encountered a tricky problem about {req.section_topic}!",
                "key_points": [f"Understand {req.section_topic}", "Practice", "Apply"],
                "examples": [],
            }


@app.post("/api/questions/generate")
async def generate_question(req: QuestionRequest):
    """Original math question via moderator."""
    from agents.moderator import SessionContext as SC
    moderator = _get_math_moderator()
    ctx = SC(
        character=req.character,
        topic=req.topic,
        target_language="en",
        difficulty=req.difficulty,
        attention_state=req.attention_state,
        question_number=req.question_number,
        total_questions=req.total_questions,
        previous_questions=req.previous_questions,
        session_accuracy=req.session_accuracy,
        last_rt=req.last_rt,
        attention_confidence=req.attention_confidence,
        eye_metrics=req.eye_metrics,
        recent_states=req.recent_states,
        last_correct=req.last_correct,
    )
    try:
        packet = moderator.run(ctx)
        return {
            "question": packet.question,
            "options": packet.options,
            "correct_index": packet.correct_index,
            "explanation": packet.explanation,
            "difficulty": packet.difficulty,
            "hints": packet.hints,
            "state_used": packet.state_used,
        }
    except Exception:
        return _math_fallback(req)


def _math_fallback(req: QuestionRequest) -> dict:
    a, b = random.randint(2, 9), random.randint(1, 12)
    x_val = random.randint(1, 10)
    c = a * x_val + b
    correct = f"x = {x_val}"
    opts_raw = [correct, f"x = {x_val+1}", f"x = {x_val-1}", f"x = {x_val+2}"]
    random.shuffle(opts_raw)
    ci = opts_raw.index(correct)
    return {
        "question": f"{req.character} needs to solve: {a}x + {b} = {c}. What is x?",
        "options": [f"{chr(65+i)}) {o}" for i, o in enumerate(opts_raw)],
        "correct_index": ci,
        "explanation": f"{a}x = {c}-{b} = {a*x_val}, so x = {x_val}.",
        "difficulty": req.difficulty,
        "hints": ["Isolate x.", f"Subtract {b} from both sides."],
        "state_used": req.attention_state,
    }


@app.post("/api/feedback/generate")
async def generate_feedback(req: FeedbackRequest):
    pct = (req.total_correct / max(1, req.total_questions)) * 100
    xp  = req.total_correct * 20 + (10 if pct >= 80 else 0)
    prompt = f"""You are {req.character}, giving feedback to a student who finished a math quiz.
Results: {req.total_correct}/{req.total_questions} ({pct:.0f}%)  Section: {req.section_title}
Attention states: {', '.join(req.attention_history[-5:]) or 'N/A'}
Give: 1) encouraging message in character voice, 2) specific tip, 3) XP = {xp}.
Return ONLY valid JSON:
{{"message": "...", "tip": "...", "xp_earned": {xp}, "rating": "excellent/good/keep_trying"}}"""
    try:
        # Use enhanced feedback client
        data = feedback_client.generate_json(prompt, expected_schema={
            "message": str,
            "tip": str,
            "xp_earned": int,
            "rating": str
        })
        data.setdefault("xp_earned", xp)
        return data
    except Exception as e:
        # Fallback to legacy method
        try:
            data = call_gemini_json(feedback_model, prompt)
            data.setdefault("xp_earned", xp)
            return data
        except Exception:
            rating = "excellent" if pct >= 80 else "good" if pct >= 50 else "keep_trying"
            return {"message": f"Great effort! {req.total_correct}/{req.total_questions}",
                    "tip": "Practice the ones you got wrong.",
                    "xp_earned": xp, "rating": rating}


@app.post("/api/evaluate/generate")
async def generate_evaluation(req: EvaluateRequest):
    prompt = f"""Generate 5 challenging math MCQ questions for a FINAL MASTERY TEST on "{req.topic}".
Character: {req.character}  Chapter: {req.chapter_title}  Difficulty: 4/5  Ages 11-15.
Return ONLY valid JSON:
{{"questions": [{{"question":"...","options":["A)...","B)...","C)...","D)..."],"correct_index":0,"explanation":"..."}}]}}"""
    try:
        # Use enhanced question client
        return question_client.generate_json(prompt, expected_schema={
            "questions": list
        })
    except Exception as e:
        # Fallback to legacy method
        try:
            return call_gemini_json(question_model, prompt)
        except Exception:
            questions = []
            for _ in range(5):
                a, x = random.randint(2, 8), random.randint(1, 8)
                b = random.randint(1, 10)
                c = a * x + b
                correct = f"x = {x}"
                opts = [correct, f"x={x+1}", f"x={x-1}", f"x={x+2}"]
                random.shuffle(opts)
                ci = opts.index(correct)
                questions.append({
                    "question": f"{req.character}: {a}x+{b}={c}. Solve for x.",
                    "options": [f"{chr(65+j)}) {o}" for j, o in enumerate(opts)],
                    "correct_index": ci, "explanation": f"x = {x}",
                })
            return {"questions": questions}


@app.post("/api/progress/update")
async def update_progress(req: ProgressUpdate):
    sid = req.student_id
    if sid not in math_progress_store:
        math_progress_store[sid] = {"sections": {}, "total_xp": 0, "history": []}
    store = math_progress_store[sid]
    sec = store["sections"].setdefault(req.section_id, {
        "answered": 0, "correct": 0, "attention_states": [],
        "best_streak": 0, "current_streak": 0,
    })
    sec["answered"] += 1
    if req.correct:
        sec["correct"] += 1
        sec["current_streak"] += 1
        sec["best_streak"] = max(sec["best_streak"], sec["current_streak"])
    else:
        sec["current_streak"] = 0
    sec["attention_states"].append(req.attention_state)
    store["total_xp"] += req.xp_earned
    store["history"].append({"section_id": req.section_id, "correct": req.correct,
                              "rt": req.response_time, "state": req.attention_state,
                              "ts": time.time()})
    return {"status": "ok", "section": sec, "total_xp": store["total_xp"]}


@app.get("/api/dashboard/{student_id}")
async def get_dashboard(student_id: str):
    store = math_progress_store.get(student_id, {"sections": {}, "total_xp": 0, "history": []})
    sections_summary = []
    for sid, sec in store["sections"].items():
        acc = (sec["correct"] / max(1, sec["answered"])) * 100
        state_counts = Counter(sec["attention_states"])
        dominant = state_counts.most_common(1)[0][0] if state_counts else "N/A"
        sections_summary.append({
            "section_id": sid, "answered": sec["answered"], "correct": sec["correct"],
            "accuracy": round(acc, 1), "best_streak": sec["best_streak"],
            "dominant_state": dominant, "state_counts": dict(state_counts),
        })
    ta = sum(s["answered"] for s in store["sections"].values())
    tc = sum(s["correct"]  for s in store["sections"].values())
    return {
        "student_id": student_id, "total_xp": store["total_xp"],
        "level": max(1, store["total_xp"] // 100 + 1),
        "total_answered": ta, "total_correct": tc,
        "overall_accuracy": round((tc / max(1, ta)) * 100, 1),
        "sections": sections_summary, "recent_history": store["history"][-20:],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Language Learning Endpoints (NEW)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/lang/languages")
async def get_languages():
    """List all supported target languages."""
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in SUPPORTED_LANGUAGES.items()
        ],
        "exercise_types": EXERCISE_TYPES,
        "skill_tags": SKILL_TAGS,
    }


@app.post("/api/lang/next_question")
async def lang_next_question(body: LangNextQuestionRequest):
    """
    Adaptive language question — full 5-agent pipeline.

    Flow:
      1. Load mastery scores for this student → feed to ReasoningAgent
      2. ReasoningAgent decides skill + difficulty + exercise type
      3. LangQuestionAgent generates the exercise
      4. StoryAgent wraps in character narrative
      5. QAAgent validates
      6. HintAgent generates 2 graduated hints
      7. Return QuestionPacket

    The response includes mastery_hint for the frontend to display spaced-rep
    guidance, and skill_tag for the progress tracker to update mastery.
    """
    moderator = _get_lang_moderator()
    eye_dict = None
    if body.eye_metrics:
        eye_dict = {k: v for k, v in body.eye_metrics.model_dump().items()
                    if v is not None}
        if not eye_dict:
            eye_dict = None

    # Pull current mastery scores for Krashen i+1 decisions
    mastery_scores = _get_mastery_scores(body.student_id)

    ctx = SessionContext(
        character=body.character,
        topic=body.topic,
        target_language=body.target_language,
        difficulty=body.difficulty,
        attention_state=body.attention_state,
        question_number=body.question_number,
        total_questions=body.total_questions,
        previous_questions=body.previous_questions[-5:],
        session_accuracy=body.session_accuracy,
        last_rt=body.last_rt,
        attention_confidence=body.attention_confidence,
        eye_metrics=eye_dict,
        recent_states=body.recent_states[-5:],
        last_correct=body.last_correct,
        skill_focus=body.skill_focus,
        mastery_scores=mastery_scores,
        exercise_type=body.exercise_type,
        learner_age=body.learner_age,
        session_fatigue=body.session_fatigue,
    )

    pkt = moderator.run(ctx)

    return {
        "question":           pkt.question,
        "options":            pkt.options,
        "correct_index":      pkt.correct_index,
        "explanation":        pkt.explanation,
        "difficulty":         pkt.difficulty,
        "hints":              pkt.hints,
        "state_used":         pkt.state_used,
        "reasoning":          pkt.reasoning or "",
        # Language-specific
        "skill_tag":          pkt.skill_tag,
        "exercise_type":      pkt.exercise_type,
        "grammar_explanation": pkt.grammar_explanation,
        "visual_breakdown":   pkt.visual_breakdown,
        "dopamine_reward":    pkt.dopamine_reward,
        "acceptable_answers": pkt.acceptable_answers,
        "native_word":        pkt.native_word,
        "target_translation": pkt.target_translation,
        "mastery_hint":       pkt.mastery_hint,
    }


@app.post("/api/lang/story/generate")
async def lang_generate_story(req: LangStoryRequest):
    """
    Generate a short immersion story in the target language.

    Research basis:
    • Krashen (1982): Comprehensible input — story is ~70% familiar + 30% new.
    • Mason & Krashen (1997): Free Voluntary Reading improves acquisition.
    • ADHD design: max 5 short paragraphs, each ≤60 words.
    """
    lang_name = SUPPORTED_LANGUAGES.get(req.target_language, req.target_language)
    prompt = f"""You are a language immersion teacher creating a short story for a {req.learner_age}-year-old
learning {lang_name}. Level: {req.difficulty}/5 (1=beginner with English translations, 5=near-native).

Character: {req.character}  |  Topic: {req.topic}

ADHD DESIGN RULES:
- Maximum 5 paragraphs, each ≤60 words.
- Beginner levels (1-2): 70% English story with target language words bolded.
- Intermediate (3): 50/50 mix. Advanced (4-5): Mostly {lang_name} with glossary.
- Comprehensible input: i+1 — introduce ≤5 new words per story.
- End with 3 "key words learned" with pronunciation guide.

Return ONLY valid JSON:
{{
  "title": "Engaging story title (bilingual ok)",
  "story": "Story text (use \\n\\n between paragraphs)",
  "key_words": [
    {{"word": "{lang_name} word", "meaning": "English meaning",
      "pronunciation": "phonetic guide", "example": "example sentence"}}
  ],
  "comprehension_question": "One simple question about the story",
  "comprehension_answer": "Answer to the question",
  "language_tip": "One short grammar or cultural tip from the story"
}}"""
    try:
        # Use enhanced story client for language stories
        return story_client.generate_json(prompt, expected_schema={
            "title": str,
            "story": str,
            "key_words": list,
            "comprehension_question": str,
            "comprehension_answer": str,
            "language_tip": str
        })
    except Exception as e:
        # Fallback to legacy method
        try:
            return call_gemini_json(story_model, prompt)
        except Exception:
            return {
                "title": f"{req.character} Learns {lang_name}",
                "story": f"{req.character} is starting to learn {lang_name}. Every day, they practice a little.\n\n"
                         f"Today's topic is: {req.topic}. Let's explore together!",
                "key_words": [{"word": "Hola", "meaning": "Hello",
                               "pronunciation": "OH-lah", "example": "Hola, ¿cómo estás?"}],
                "comprehension_question": f"What is {req.character} learning today?",
                "comprehension_answer": req.topic,
                "language_tip": f"Practice {req.topic} every day for best results!",
            }


@app.post("/api/lang/feedback/generate")
async def lang_generate_feedback(req: LangFeedbackRequest):
    """
    End-of-session language feedback.

    Includes:
    • Session performance summary
    • Skill-specific mastery advice
    • Spaced-repetition recommendation (which skills to review next)
    • Shame-free, ADHD-optimised tone

    Research: Lyster & Ranta (1997) — recast feedback (showing correct form
    in response) is more effective than explicit error correction for acquisition.
    """
    lang_name = SUPPORTED_LANGUAGES.get(req.target_language, req.target_language)
    pct = (req.total_correct / max(1, req.total_questions)) * 100
    xp  = req.total_correct * 25 + (15 if pct >= 80 else 0)

    # Mastery advice from skill results
    weak_skills  = [s for s, acc in req.skill_results.items() if acc < 0.5]
    strong_skills = [s for s, acc in req.skill_results.items() if acc >= 0.8]

    prompt = f"""You are {req.character}, a warm and encouraging language tutor speaking to a
{lang_name} learner with ADHD who just finished a practice session.

Session results: {req.total_correct}/{req.total_questions} correct ({pct:.0f}%)
Topic: {req.topic}  |  Attention pattern: {', '.join(req.attention_history[-5:]) or 'N/A'}
Strong skills: {', '.join(strong_skills) or 'building up'}
Skills needing practice: {', '.join(weak_skills) or 'none — great work!'}
XP earned: {xp}

TONE RULES (ADHD research — Lam & Muldner 2018):
- Celebrate effort first, achievement second
- NEVER say "wrong", "failed", "bad" — always recast positively
- Keep each message ≤3 sentences
- Give ONE concrete next-step tip only

Return ONLY valid JSON:
{{
  "message": "Warm, energetic in-character message (≤3 sentences)",
  "skill_note": "One positive recast note about a weak skill (≤2 sentences)",
  "next_step": "One concrete micro-action for next session",
  "review_skills": {json.dumps(weak_skills)},
  "xp_earned": {xp},
  "rating": "{('excellent' if pct >= 80 else 'good' if pct >= 50 else 'keep_going')}",
  "encouragement_quote": "A short motivational quote about language learning"
}}"""

    try:
        # Use enhanced feedback client for language feedback
        data = feedback_client.generate_json(prompt, expected_schema={
            "message": str,
            "skill_note": str,
            "next_step": str,
            "review_skills": list,
            "xp_earned": int,
            "rating": str,
            "encouragement_quote": str
        })
        data.setdefault("xp_earned", xp)
        return data
    except Exception as e:
        # Fallback to legacy method
        try:
            data = call_gemini_json(feedback_model, prompt)
            data.setdefault("xp_earned", xp)
            return data
        except Exception:
            rating = "excellent" if pct >= 80 else "good" if pct >= 50 else "keep_going"
            return {
                "message": f"Amazing effort today! {req.total_correct} out of {req.total_questions} — you're building something real!",
                "skill_note": f"Keep practising {weak_skills[0] if weak_skills else 'all your skills'} — it gets easier every time.",
                "next_step": f"Try 5 minutes of {lang_name} tomorrow to lock in today's words.",
                "review_skills": weak_skills,
                "xp_earned": xp,
                "rating": rating,
                "encouragement_quote": "Every expert was once a beginner.",
            }


@app.post("/api/lang/progress/update")
async def lang_update_progress(req: LangProgressUpdate):
    """
    Update language progress after each answer.

    Does three things:
    1. Updates session-level stats (accuracy, streak, XP)
    2. Updates EWMA mastery score for the skill tag
    3. Runs SM-2 spaced-repetition scheduler for next review

    The returned mastery_scores dict is fed back into the next
    /api/lang/next_question call so ReasoningAgent always has fresh data.
    """
    sid = req.student_id
    if sid not in lang_progress_store:
        lang_progress_store[sid] = {
            "sections": {}, "total_xp": 0,
            "history": [], "languages": defaultdict(dict),
        }

    store = lang_progress_store[sid]
    sec = store["sections"].setdefault(req.section_id, {
        "answered": 0, "correct": 0, "attention_states": [],
        "best_streak": 0, "current_streak": 0,
        "exercise_types": Counter(), "skill_tags": Counter(),
    })

    sec["answered"] += 1
    if req.correct:
        sec["correct"] += 1
        sec["current_streak"] += 1
        sec["best_streak"] = max(sec["best_streak"], sec["current_streak"])
    else:
        sec["current_streak"] = 0

    sec["attention_states"].append(req.attention_state)
    sec["exercise_types"][req.exercise_type] += 1
    sec["skill_tags"][req.skill_tag] += 1
    store["total_xp"] += req.xp_earned

    store["history"].append({
        "section_id":   req.section_id,
        "skill_tag":    req.skill_tag,
        "exercise_type": req.exercise_type,
        "correct":      req.correct,
        "rt":           req.response_time,
        "state":        req.attention_state,
        "language":     req.target_language,
        "ts":           time.time(),
    })

    # Update mastery + spaced repetition
    mastery_data = _update_mastery(
        sid, req.skill_tag, req.correct, req.response_time
    )

    return {
        "status":        "ok",
        "section":       {
            "answered":       sec["answered"],
            "correct":        sec["correct"],
            "accuracy":       round(sec["correct"] / max(1, sec["answered"]) * 100, 1),
            "current_streak": sec["current_streak"],
            "best_streak":    sec["best_streak"],
        },
        "total_xp":      store["total_xp"],
        "skill_mastery": mastery_data["mastery"],
        "next_review_in_days": mastery_data["interval_days"],
        "mastery_scores": _get_mastery_scores(sid),
    }


@app.get("/api/lang/mastery/{student_id}")
async def get_mastery(student_id: str):
    """
    Return full mastery state for all skills.
    Used by the LSTM training pipeline and the frontend skill radar chart.
    """
    if student_id not in mastery_store:
        return {"student_id": student_id, "skills": {}, "total_interactions": 0}

    skills = mastery_store[student_id]
    now = time.time()
    due_for_review = [
        tag for tag, data in skills.items()
        if data.get("next_review_ts", 0) <= now
    ]

    return {
        "student_id":      student_id,
        "skills":          skills,
        "due_for_review":  due_for_review,
        "total_interactions": sum(s.get("interactions", 0) for s in skills.values()),
        "mastery_vector":  _get_mastery_scores(student_id),
    }


@app.get("/api/lang/dashboard/{student_id}")
async def lang_get_dashboard(student_id: str):
    """
    Language learning dashboard.

    Returns:
    • Per-section stats (accuracy, streaks, attention states)
    • Per-skill mastery scores and SM-2 review schedule
    • XP, level, and engagement metrics
    • Attention state distribution (for ADHD progress monitoring)
    """
    store = lang_progress_store.get(student_id, {
        "sections": {}, "total_xp": 0, "history": []
    })

    sections_summary = []
    for sid, sec in store["sections"].items():
        acc = (sec["correct"] / max(1, sec["answered"])) * 100
        state_counts = Counter(sec.get("attention_states", []))
        dominant = state_counts.most_common(1)[0][0] if state_counts else "N/A"
        sections_summary.append({
            "section_id":     sid,
            "answered":       sec["answered"],
            "correct":        sec["correct"],
            "accuracy":       round(acc, 1),
            "best_streak":    sec["best_streak"],
            "dominant_state": dominant,
            "state_counts":   dict(state_counts),
            "exercise_types": dict(sec.get("exercise_types", {})),
        })

    ta = sum(s["answered"] for s in store["sections"].values())
    tc = sum(s["correct"]  for s in store["sections"].values())

    # Mastery radar for frontend chart
    mastery_scores = _get_mastery_scores(student_id)

    # Recent attention trend
    recent_hist = store["history"][-50:]
    attention_trend = Counter(h["state"] for h in recent_hist)

    total_xp  = store.get("total_xp", 0)
    level     = max(1, total_xp // 150 + 1)
    xp_to_next = level * 150 - total_xp

    return {
        "student_id":       student_id,
        "total_xp":         total_xp,
        "level":            level,
        "xp_to_next_level": xp_to_next,
        "total_answered":   ta,
        "total_correct":    tc,
        "overall_accuracy": round((tc / max(1, ta)) * 100, 1),
        "sections":         sections_summary,
        "mastery_scores":   mastery_scores,
        "attention_distribution": dict(attention_trend),
        "skills_due_for_review": [
            tag for tag, data in mastery_store.get(student_id, {}).items()
            if data.get("next_review_ts", 0) <= time.time()
        ],
        "recent_history":   recent_hist[-20:],
        # ADHD progress insight
        "adhd_insights": _compute_adhd_insights(recent_hist, mastery_scores),
    }


def _compute_adhd_insights(history: List[Dict], mastery: Dict[str, float]) -> Dict:
    """
    Compute ADHD-relevant learning analytics.

    Research: Toplak et al. (2013) — time-on-task degradation in ADHD.
    Monitors attention state drift over session and accuracy trends.
    """
    if not history:
        return {"focus_rate": 0, "best_time_of_session": "start", "streak_potential": 0}

    states = [h.get("state", "Focused") for h in history]
    focus_rate = round(states.count("Focused") / len(states) * 100, 1)

    # Accuracy in first vs second half of session
    mid = len(history) // 2
    first_half_acc = sum(1 for h in history[:mid] if h.get("correct")) / max(1, mid)
    second_half_acc = sum(1 for h in history[mid:] if h.get("correct")) / max(1, len(history) - mid)
    best_half = "start" if first_half_acc >= second_half_acc else "end"

    # Average mastery
    avg_mastery = sum(mastery.values()) / max(1, len(mastery))

    return {
        "focus_rate":          focus_rate,
        "best_time_of_session": best_half,
        "first_half_accuracy":  round(first_half_acc * 100, 1),
        "second_half_accuracy": round(second_half_acc * 100, 1),
        "avg_mastery":          round(avg_mastery * 100, 1),
        "fatigue_detected":     second_half_acc < first_half_acc - 0.15,
        "recommendation":       (
            "Consider ending the session — attention is drifting."
            if second_half_acc < first_half_acc - 0.15
            else "Great sustained focus! Keep going."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NLP Validation Endpoint (NEW)
# ══════════════════════════════════════════════════════════════════════════════

class AnswerValidationRequest(BaseModel):
    """Request for answer validation using NLP model."""
    learner_answer: str
    correct_answer: str
    acceptable_answers: List[str] = []
    target_language: str = "es"


@app.post("/api/lang/validate_answer")
async def validate_answer(req: AnswerValidationRequest):
    """
    Validate learner answers using the NLP model.
    
    Uses semantic similarity, edit distance, and error classification
    to provide flexible answer validation with detailed feedback.
    """
    nlp_model = _get_nlp_model()
    
    try:
        result = nlp_model.validate_answer(
            learner_answer=req.learner_answer,
            correct_answer=req.correct_answer,
            acceptable_answers=req.acceptable_answers,
            target_language=req.target_language,
        )
        return {
            "success": True,
            "validation": result
        }
    except Exception as e:
        raise HTTPException(500, f"NLP validation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
