"""
agents/moderator.py  — NeuroLearn Language Edition
====================================================
Multi-agent orchestrator for adaptive language teaching.

Expanded from the original math pipeline to support:
  • Vocabulary, Grammar, Reading Comprehension, Translation, Listening (text)
  • ADHD-optimised micro-chunks: max 25-word instructions, visual breakdowns,
    semantic answer acceptance, shame-free feedback
  • Full attention-aware routing: Focused / Drifting / Impulsive / Overwhelmed
  • Spaced-repetition metadata per exercise for LSTM mastery scheduling

═══════════════════════════════════════════════════════════════════════════════
Research basis
───────────────────────────────────────────────────────────────────────────────
• Krashen (1982) — Comprehensible Input Hypothesis:
    Language acquisition happens when input is i+1 (just above current level).
    → difficulty is dynamically adjusted by ReasoningAgent based on attention.

• Swain (1985) — Output Hypothesis:
    Producing language (not just consuming it) strengthens acquisition.
    → every exercise asks learners to produce/choose output, not just read.

• Nation (2001) — Vocabulary learning:
    Spaced, varied encounters (reading/writing/speaking contexts) build retention.
    → exercises rotate context even for the same word.

• Lyster & Ranta (1997) — Corrective Feedback:
    Recasting errors without shame is more effective than explicit correction.
    → QA/Feedback agents always recast, never shame.

• Mayer (2009) — Multimedia Learning Theory:
    Dual-channel (verbal + visual) processing aids retention.
    → visual_breakdown field in every packet (emoji / colour-coded grammar).

• Baddeley & Hitch (1974) — Working Memory Model:
    Central executive overload → forgetting. ADHD students have reduced WM.
    → Overwhelmed state triggers single-step, minimal-text exercises.

• Toplak et al. (2013) — ADHD executive function & time-on-task:
    Performance degrades faster in ADHD. Micro-sessions (3-5 min) preserve it.
    → session_fatigue tracked; lesson ends or switches modality at threshold.

• Lam & Muldner (2018) — Gamification & ADHD:
    Immediate, surprising rewards activate dopamine circuits more than deferred
    XP. → dopamine_reward string delivered synchronously with correct answer.

Pipeline (per question)
───────────────────────
  0. ReasoningAgent  — reads fused attention + eye data → strategy
  1. LangQuestionAgent — generates raw exercise (vocab / grammar / fill / translate)
  2. StoryAgent      — wraps in narrative context using the chosen character
  3. QAAgent         — validates linguistic correctness + ADHD-safety
     └── retry up to MAX_RETRIES
  4. HintAgent       — pre-generates 2 graduated, recast hints
  → QuestionPacket returned to app.py
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from gemini_utils import (
    get_gemini_client,
    call_gemini_json as enhanced_call_gemini_json,
    GeminiConfig,
    JSONParser
)

# ─────────────────────────────────────────────────────────────
# Supported languages & their native names
# ─────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "zh": "Mandarin Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "ko": "Korean",
}

# ─────────────────────────────────────────────────────────────
# Exercise type catalogue
# ─────────────────────────────────────────────────────────────
EXERCISE_TYPES = [
    "multiple_choice_vocab",   # pick the right translation / meaning
    "fill_in_the_blank",       # complete the sentence with the right word/form
    "translation",             # translate a short phrase EN → target
    "grammar_sort",            # drag-word / pick correct grammar form
    "listening_text",          # read aloud text, answer comprehension MCQ
    "match_pairs",             # match word ↔ meaning (displayed as MCQ)
    "sentence_builder",        # arrange words into correct order (MCQ)
]

# ─────────────────────────────────────────────────────────────
# Skill tags (for LSTM mastery tracking)
# ─────────────────────────────────────────────────────────────
SKILL_TAGS = {
    "vocabulary_basic": "Core everyday words",
    "vocabulary_intermediate": "Context-dependent vocabulary",
    "grammar_present": "Present tense conjugation",
    "grammar_past": "Past tense / preterite",
    "grammar_future": "Future tense",
    "grammar_genders": "Noun gender & articles",
    "grammar_plurals": "Plural formation",
    "grammar_adjectives": "Adjective agreement",
    "grammar_pronouns": "Personal & object pronouns",
    "reading_comprehension": "Short passage understanding",
    "translation_basic": "Simple phrase translation",
    "translation_intermediate": "Complex sentence translation",
    "listening_comprehension": "Audio/text listening comprehension",
    "sentence_structure": "Word order & syntax",
    "numbers_time": "Numbers, dates, times",
    "social_phrases": "Greetings, politeness, everyday phrases",
}


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class EyeMetricsPayload:
    blink_rate:        Optional[float] = None
    pupil_dilation:    Optional[float] = None
    fixation_duration: Optional[float] = None
    saccade_rate:      Optional[float] = None
    gaze_stability:    Optional[float] = None


@dataclass
class QuestionPacket:
    """Complete, validated output of one moderator cycle."""
    question:          str
    options:           List[str]        # ["A) ...", "B) ...", "C) ...", "D) ..."]
    correct_index:     int              # 0-3
    explanation:       str
    difficulty:        int              # 1-5
    hints:             List[str]        # [easy_hint, harder_hint]
    state_used:        str              # attention state
    qa_passed:         bool
    attempts:          int
    # ── Language-specific fields ───────────────────────────
    skill_tag:         str = ""                    # for LSTM mastery tracking
    exercise_type:     str = "multiple_choice_vocab"
    grammar_explanation: str = ""                 # concise grammar rule
    visual_breakdown:  str = ""                   # emoji / colour-coded pattern
    dopamine_reward:   str = "🌟 Great job!"     # instant positive feedback
    acceptable_answers: List[str] = field(default_factory=list)  # semantic variants
    native_word:       str = ""                   # word being taught
    target_translation: str = ""                  # correct translation
    mastery_hint:      str = ""                   # spaced-rep scheduling note
    # ── Meta ──────────────────────────────────────────────
    reasoning:         Optional[str] = None
    agent_log:         List[str] = field(default_factory=list)


@dataclass
class SessionContext:
    """Shared state passed to all agents each call."""
    character:             str            # chosen avatar / character name
    topic:                 str            # e.g. "Spanish greetings"
    target_language:       str            # ISO code: "es", "fr", etc.
    difficulty:            int            # 1-5
    attention_state:       str            # Focused / Drifting / Impulsive / Overwhelmed
    question_number:       int
    total_questions:       int
    previous_questions:    List[str]      # avoid repetition
    session_accuracy:      float          # running 0-1
    # ── Attention signals ─────────────────────────────────
    last_rt:               Optional[float] = None
    attention_confidence:  float = 0.5
    eye_metrics:           Optional[Dict] = None
    recent_states:         List[str] = field(default_factory=list)
    last_correct:          Optional[bool] = None
    # ── Language-specific context ─────────────────────────
    skill_focus:           str = "vocabulary_basic"   # current skill tag
    mastery_scores:        Dict[str, float] = field(default_factory=dict)
    exercise_type:         Optional[str] = None   # forced type (optional)
    learner_age:           int = 12               # default ADHD target age
    session_fatigue:       float = 0.0            # 0-1, rises per question


# ─────────────────────────────────────────────────────────────
# Moderator — orchestrates all agents
# ─────────────────────────────────────────────────────────────

class Moderator:
    """
    Language-teaching moderator.

    Agents:
        ReasoningAgent    → reads attention + mastery → strategy decision
        LangQuestionAgent → generates the raw exercise
        StoryAgent        → wraps in character narrative
        QAAgent           → validates linguistic accuracy + ADHD safety
        HintAgent         → generates 2 graduated hints (recast style)
    """

    MAX_RETRIES = 2

    def __init__(self, api_key: str, verbose: bool = False):
        # Use enhanced Gemini configuration
        self.config = GeminiConfig(api_key=api_key)
        self.client = get_gemini_client(self.config)
        self._verbose = verbose
        
        # Legacy model instances for backward compatibility
        genai.configure(api_key=api_key)
        self._reasoning_agent  = _make_model(temperature=0.2)
        self._question_agent   = _make_model(temperature=0.75)
        self._story_agent      = _make_model(temperature=0.9)
        self._qa_agent         = _make_model(temperature=0.3)
        self._hint_agent       = _make_model(temperature=0.5)

    def run(self, ctx: SessionContext) -> QuestionPacket:
        log: List[str] = []

        # ── Step 0: ReasoningAgent ────────────────────────────────
        reasoning_result   = _call_reasoning_agent(self._reasoning_agent, ctx)
        reasoning_text     = reasoning_result.get("reasoning", "")
        suggested_diff     = reasoning_result.get("suggested_difficulty")
        suggested_guidance = reasoning_result.get("suggested_guidance")
        suggested_tone     = reasoning_result.get("suggested_tone")
        suggested_skill    = reasoning_result.get("suggested_skill")
        suggested_type     = reasoning_result.get("suggested_exercise_type")
        dopamine_msg       = reasoning_result.get("dopamine_reward", "🌟 You got this!")

        # Apply reasoning overrides
        if suggested_diff is not None:
            ctx = _replace_ctx(ctx, difficulty=min(5, max(1, int(suggested_diff))))
        if suggested_skill:
            ctx = _replace_ctx(ctx, skill_focus=suggested_skill)
        if suggested_type:
            ctx = _replace_ctx(ctx, exercise_type=suggested_type)

        log.append(f"[ReasoningAgent] {reasoning_text[:80]}")

        for attempt in range(1, self.MAX_RETRIES + 2):
            log.append(f"=== Attempt {attempt} ===")
            try:
                # ── Step 1: LangQuestionAgent ─────────────────────
                raw = _call_question_agent(
                    self._question_agent, ctx, suggested_guidance
                )
                log.append(f"[LangQuestionAgent] type={raw.get('exercise_type','?')} "
                            f"skill={raw.get('skill_tag','?')}")

                # ── Step 2: StoryAgent ────────────────────────────
                story = _call_story_agent(
                    self._story_agent, ctx, raw, suggested_tone, dopamine_msg
                )
                log.append(f"[StoryAgent] q={story.get('question','?')[:60]}")

                # ── Step 3: QAAgent ───────────────────────────────
                qa = _call_qa_agent(self._qa_agent, ctx, story, raw)
                log.append(f"[QAAgent] passed={qa['passed']} "
                            f"reason={qa.get('reason','')[:50]}")

                if qa["passed"] or attempt == self.MAX_RETRIES + 1:
                    # ── Step 4: HintAgent ─────────────────────────
                    hints = _call_hint_agent(self._hint_agent, ctx, story, raw)
                    log.append("[HintAgent] hints ready")

                    if self._verbose:
                        for line in log:
                            print(f"  {line}")

                    return QuestionPacket(
                        question          = story["question"],
                        options           = story["options"],
                        correct_index     = story["correct_index"],
                        explanation       = story.get("explanation", ""),
                        difficulty        = story.get("difficulty", ctx.difficulty),
                        hints             = hints,
                        state_used        = ctx.attention_state,
                        qa_passed         = qa["passed"],
                        attempts          = attempt,
                        skill_tag         = raw.get("skill_tag", ctx.skill_focus),
                        exercise_type     = raw.get("exercise_type", "multiple_choice_vocab"),
                        grammar_explanation = raw.get("grammar_explanation", ""),
                        visual_breakdown  = raw.get("visual_breakdown", ""),
                        dopamine_reward   = story.get("dopamine_reward", dopamine_msg),
                        acceptable_answers = raw.get("acceptable_answers", []),
                        native_word       = raw.get("native_word", ""),
                        target_translation = raw.get("target_translation", ""),
                        mastery_hint      = raw.get("mastery_hint", ""),
                        reasoning         = reasoning_text,
                        agent_log         = log,
                    )

            except Exception as e:
                log.append(f"[ERROR] attempt {attempt}: {e}")
                if self._verbose:
                    print(f"  [Moderator] Error on attempt {attempt}: {e}")

        # ── Fallback ──────────────────────────────────────────────
        log.append("[Moderator] Using fallback.")
        fb = _build_fallback(ctx)
        return QuestionPacket(**fb, state_used=ctx.attention_state,
                               qa_passed=False, attempts=self.MAX_RETRIES + 1,
                               agent_log=log, reasoning=reasoning_text)


# ─────────────────────────────────────────────────────────────
# Agent implementations
# ─────────────────────────────────────────────────────────────

def _call_reasoning_agent(model, ctx: SessionContext) -> Dict:
    """
    ReasoningAgent.

    Uses fused attention state + eye metrics + mastery scores to decide:
      - Should we reinforce (mastery < 0.4), extend (0.4-0.7), or advance (>0.7)?
      - What exercise type fits this attention state?
      - What difficulty adjustment is needed?

    Research: Vygotsky's Zone of Proximal Development — always aim i+1 above
    current mastery, but only when attention allows it. Under Overwhelmed/Drifting,
    drop to i or even i-1.
    """
    lang_name = SUPPORTED_LANGUAGES.get(ctx.target_language, ctx.target_language)

    # Mastery summary
    mastery_str = ""
    if ctx.mastery_scores:
        lines = [f"  {k}: {v:.0%}" for k, v in ctx.mastery_scores.items()]
        mastery_str = "Skill mastery scores:\n" + "\n".join(lines)

    # Eye data
    eye_block = ""
    if ctx.eye_metrics:
        m = ctx.eye_metrics
        eye_block = (
            f"\nEYE-TRACKING:\n"
            f"  blink_rate={m.get('blink_rate',0):.1f}/min  "
            f"pupil_dilation={m.get('pupil_dilation',0):+.1f}%  "
            f"fixation={m.get('fixation_duration',0):.0f}ms  "
            f"saccade={m.get('saccade_rate',0):.1f}/s  "
            f"gaze_stability={m.get('gaze_stability',1):.2f}"
        )

    recent = ", ".join(ctx.recent_states[-5:]) or "none"
    last_outcome = (
        "correct" if ctx.last_correct
        else "incorrect" if ctx.last_correct is False
        else "first question"
    )

    prompt = f"""You are the REASONING AGENT in an adaptive LANGUAGE tutoring system for ADHD learners.

RESEARCH BRIEF:
- Krashen (1982): target i+1 — just above current mastery when attention is good.
- Vygotsky ZPD: drop to reinforcement level when attention is low.
- ADHD cognitive load: reduce task complexity under Overwhelmed/Drifting states.
- Lam & Muldner (2018): immediate dopamine rewards beat deferred XP.

SESSION STATE:
  Language: {lang_name}  |  Topic: {ctx.topic}  |  Skill: {ctx.skill_focus}
  Difficulty: {ctx.difficulty}/5  |  Q{ctx.question_number}/{ctx.total_questions}
  Attention state: {ctx.attention_state} (conf: {ctx.attention_confidence:.0%})
  Recent states: {recent}
  Last answer: {last_outcome}  |  Last RT: {ctx.last_rt}s
  Session accuracy: {ctx.session_accuracy:.0%}
  Session fatigue: {ctx.session_fatigue:.0%}
  Learner age: {ctx.learner_age}
{mastery_str}{eye_block}

EXERCISE TYPES AVAILABLE:
{chr(10).join(f"  - {t}" for t in EXERCISE_TYPES)}

SKILL TAGS AVAILABLE:
{chr(10).join(f"  - {k}: {v}" for k,v in SKILL_TAGS.items())}

YOUR JOB:
Decide the optimal next exercise strategy. Return ONLY valid JSON:
{{
  "reasoning": "1-2 sentences explaining your strategy choice.",
  "suggested_difficulty": null,
  "suggested_skill": "{ctx.skill_focus}",
  "suggested_exercise_type": "multiple_choice_vocab",
  "suggested_guidance": "Specific instruction for the question agent.",
  "suggested_tone": "Specific tone instruction for the story agent.",
  "dopamine_reward": "Short (≤8 words) celebratory message with emoji 🎉"
}}

RULES:
- suggested_difficulty: null to keep current, or 1-5 integer to override
- For Overwhelmed: drop difficulty -1, pick simplest exercise type, minimal text
- For Drifting: keep same difficulty but add novelty/surprise in the topic
- For Impulsive: add a deliberate trap option, flag it in suggested_guidance
- For Focused: push to harder type or advance the skill tag
- Prefer fill_in_the_blank and grammar_sort for grammar skills
- Prefer multiple_choice_vocab and match_pairs for vocabulary skills
- dopamine_reward must be positive, specific, shame-free, ≤8 words"""

    try:
        return _call_json(model, prompt)
    except Exception as e:
        print(f"[ReasoningAgent] failed: {e}")
        return {
            "reasoning": f"Defaulting for {ctx.attention_state}.",
            "suggested_difficulty": None,
            "suggested_skill": ctx.skill_focus,
            "suggested_exercise_type": "multiple_choice_vocab",
            "suggested_guidance": None,
            "suggested_tone": None,
            "dopamine_reward": "🌟 Keep going!",
        }


def _call_question_agent(
    model, ctx: SessionContext, suggested_guidance: Optional[str] = None
) -> Dict:
    """
    LangQuestionAgent.

    Generates the raw language exercise. Does NOT add story wrapper yet.
    Produces: question_stem, correct_answer, distractors, grammar_explanation,
              visual_breakdown, skill_tag, exercise_type, acceptable_answers.

    Research: Nation (2001) — vocabulary needs varied encounters in context.
    Lyster & Ranta (1997) — provide correct form implicitly (recast), not explicitly.
    """
    lang_name = SUPPORTED_LANGUAGES.get(ctx.target_language, ctx.target_language)

    state_guidance = {
        "Focused":     "multi-part exercise that requires active recall or production",
        "Drifting":    "single vivid word with a surprising or colourful example sentence",
        "Impulsive":   "exercise with a plausible trap answer (common learner mistake)",
        "Overwhelmed": "single-word / single-step exercise, tiny instruction (≤15 words)",
    }
    guidance = suggested_guidance or state_guidance.get(ctx.attention_state, "standard vocabulary MCQ")
    ex_type  = ctx.exercise_type or "multiple_choice_vocab"

    prev_block = ""
    if ctx.previous_questions:
        prev_block = "\nAvoid repeating these previous questions:\n" + \
                     "\n".join(f"  - {q}" for q in ctx.previous_questions[-5:])

    prompt = f"""You are the LANGUAGE QUESTION AGENT in an adaptive tutoring system for ADHD learners (age {ctx.learner_age}).

TARGET LANGUAGE: {lang_name} ({ctx.target_language})
TOPIC: {ctx.topic}
SKILL: {ctx.skill_focus}
EXERCISE TYPE: {ex_type}
DIFFICULTY: {ctx.difficulty}/5  (1=absolute beginner, 5=near-native)
ATTENTION STATE: {ctx.attention_state}
GUIDANCE: {guidance}
{prev_block}

EXERCISE TYPE DEFINITIONS:
- multiple_choice_vocab: "What does X mean?" or "How do you say Y in {lang_name}?"
- fill_in_the_blank: Sentence with one word missing, 4 options
- translation: "Translate: [short phrase]" into {lang_name}
- grammar_sort: Choose correct conjugation / gender / plural form
- listening_text: Short text to read, then comprehension MCQ
- match_pairs: Presented as MCQ — "X matches with?"
- sentence_builder: 4 word-order options, choose correct one

CRITICAL RULES:
1. Keep instructions under 25 words (ADHD working memory limit).
2. For Overwhelmed: question stem ≤ 12 words, 1 operation only.
3. Accept semantic variations — if the learner is right in meaning, flag it.
4. No shame in distractors — they should be plausible mistakes, not insults.
5. Grammar explanation must be ≤ 30 words, concrete, with example.
6. visual_breakdown: use emoji + colour-coded pattern (e.g. "🟦 verb + 🟨 subject").
7. Always include at least 2 acceptable_answers (e.g. informal / formal variants).

Return ONLY valid JSON:
{{
  "exercise_type": "{ex_type}",
  "skill_tag": "<one of the skill tags above>",
  "difficulty_actual": {ctx.difficulty},
  "question_stem": "The question (≤25 words for instruction part)",
  "correct_answer": "The single canonical correct answer",
  "distractors": ["wrong1", "wrong2", "wrong3"],
  "acceptable_answers": ["correct_answer", "variant1", "variant2"],
  "native_word": "The English source word or phrase",
  "target_translation": "The {lang_name} translation",
  "grammar_explanation": "≤30 word rule with mini-example",
  "visual_breakdown": "Emoji pattern e.g. 🟦I 🟨eat 🟩apples",
  "mastery_hint": "One spaced-rep note e.g. 'Review in 2 days if score < 70%'",
  "operation": "e.g. vocabulary_recall / conjugation / word_order"
}}"""

    return _call_json(model, prompt)


def _call_story_agent(
    model,
    ctx: SessionContext,
    raw: Dict,
    suggested_tone: Optional[str] = None,
    dopamine_msg: str = "🌟 Nice!",
) -> Dict:
    """
    StoryAgent.

    Wraps the raw exercise in a short character-driven narrative context.
    Research: Mayer (2009) — narrative context activates dual-channel processing
    and reduces extraneous cognitive load for ADHD learners.

    Does NOT change the linguistic content — only adds narrative wrapper.
    """
    lang_name = SUPPORTED_LANGUAGES.get(ctx.target_language, ctx.target_language)

    state_tone = {
        "Focused":     "adventurous, one engaging scene, moderate length",
        "Drifting":    "funny and punchy — one sentence opener, grab attention NOW",
        "Impulsive":   "start with 'Look carefully:' — slightly tricky narrative hook",
        "Overwhelmed": "warm, calming, 1 sentence only — like a gentle friend asking",
    }
    tone = suggested_tone or state_tone.get(ctx.attention_state, "friendly and brief")

    prompt = f"""You are the STORY AGENT in an adaptive language tutoring system.

CHARACTER: {ctx.character}
LANGUAGE: {lang_name}
TOPIC: {ctx.topic}
TONE: {tone}
ADHD RULE: Keep the question prompt under 40 words total. No long paragraphs.

RAW EXERCISE:
  type: {raw.get('exercise_type', 'multiple_choice_vocab')}
  stem: {raw.get('question_stem', '')}
  correct: {raw.get('correct_answer', '')}
  distractors: {raw.get('distractors', [])}
  explanation: {raw.get('grammar_explanation', '')}
  visual_breakdown: {raw.get('visual_breakdown', '')}

YOUR JOB:
1. Write a SHORT narrative wrapper featuring {ctx.character} — set the scene in ≤2 sentences.
2. Then present the exercise stem clearly.
3. Format 4 answer options (A–D), shuffle so correct is NOT always A.
4. correct_index: 0=A, 1=B, 2=C, 3=D — MUST match where you placed the correct answer.
5. Explanation: one warm sentence recasting the correct form (Lyster & Ranta, 1997 style).

SHAME-FREE RULE: explanation must NEVER say "wrong" or "incorrect" — only show the correct pattern.

Return ONLY valid JSON:
{{
  "question": "Narrative scene + exercise stem (≤40 words total)",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct_index": 0,
  "explanation": "Warm recast explanation",
  "difficulty": {ctx.difficulty},
  "dopamine_reward": "{dopamine_msg}",
  "visual_breakdown": "{raw.get('visual_breakdown', '')}"
}}"""

    return _call_json(model, prompt)


def _call_qa_agent(model, ctx: SessionContext, story: Dict, raw: Dict) -> Dict:
    """
    QAAgent.

    Validates:
      1. Linguistic correctness of the exercise
      2. correct_index truly points to the right option
      3. ADHD safety (no shame, no excessive length)
      4. Grammar explanation accuracy
      5. Exercise matches the skill tag

    Temperature = 0.05 for near-deterministic validation.
    """
    lang_name = SUPPORTED_LANGUAGES.get(ctx.target_language, ctx.target_language)

    prompt = f"""You are the QA AGENT for a {lang_name} language learning system targeting ADHD learners.

Be strict but constructive. Validate the following exercise:

EXERCISE:
{json.dumps(story, indent=2, ensure_ascii=False)}

RAW METADATA:
  skill_tag: {raw.get('skill_tag', '')}
  grammar_explanation: {raw.get('grammar_explanation', '')}
  acceptable_answers: {raw.get('acceptable_answers', [])}

CHECK ALL OF THESE:
1. Is the {lang_name} linguistically correct? (spelling, grammar, accents)
2. Does correct_index (0=A,1=B,2=C,3=D) ACTUALLY match the correct answer in options?
3. Are distractors plausible {lang_name} errors (not gibberish)?
4. Is the question ≤40 words? (ADHD rule)
5. Is the explanation shame-free? (no "wrong", "incorrect", "bad")
6. Is the grammar_explanation ≤30 words?
7. Does the exercise match skill tag "{raw.get('skill_tag', '')}"?
8. Are there at least 2 acceptable_answers listed?

Return ONLY valid JSON:
{{
  "passed": true,
  "reason": "Brief reason if failed, empty string if passed",
  "suggested_fix": null
}}"""

    result = _call_json(model, prompt)
    result.setdefault("passed", True)
    result.setdefault("reason", "")
    result.setdefault("suggested_fix", None)
    return result


def _call_hint_agent(
    model, ctx: SessionContext, story: Dict, raw: Dict
) -> List[str]:
    """
    HintAgent.

    Generates 2 graduated hints. Never reveals the answer.
    Style: recast (Lyster & Ranta, 1997) — show the pattern, not the answer.
    """
    lang_name = SUPPORTED_LANGUAGES.get(ctx.target_language, ctx.target_language)

    prompt = f"""You are the HINT AGENT for a {lang_name} language tutor for ADHD learners.

Generate 2 hints for a student who is stuck. NEVER reveal the answer.

QUESTION: {story.get('question', '')}
CORRECT OPTION: {story['options'][story.get('correct_index', 0)] if story.get('options') else ''}
GRAMMAR NOTE: {raw.get('grammar_explanation', '')}
VISUAL BREAKDOWN: {raw.get('visual_breakdown', '')}
CHARACTER: {ctx.character}
ATTENTION STATE: {ctx.attention_state}

HINT RULES (Lyster & Ranta recast style):
- Hint 1: Very gentle — point to the CATEGORY (e.g. "Think about the verb ending...")
- Hint 2: Show the PATTERN without the answer (e.g. "In {lang_name}, -ar verbs end in -o for 'I'...")
- For Overwhelmed: Hint 1 must be extra warm and in ≤10 words
- Reference {ctx.character} in one of the hints if natural
- Max 20 words per hint

Return ONLY valid JSON:
{{
  "hint_1": "Gentle category hint",
  "hint_2": "Pattern hint (no answer)"
}}"""

    try:
        result = _call_json(model, prompt)
        return [
            result.get("hint_1", "Think about what type of word fits here."),
            result.get("hint_2", f"In {lang_name}, look at the pattern shown."),
        ]
    except Exception:
        return [
            "Think about what type of word fits here.",
            f"Look at the pattern in the grammar note.",
        ]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_model(temperature: float = 0.7):
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(temperature=temperature),
    )


def _call_json(model, prompt: str) -> Dict:
    """Enhanced JSON call with robust parsing and fallback."""
    try:
        # Try enhanced client first if available
        if hasattr(model, 'generate_content'):
            # Legacy path - use original implementation with enhanced parsing
            resp = model.generate_content(prompt)
            text = (resp.text or "").strip()
            
            # Use enhanced JSON parser
            json_text = JSONParser.extract_json_from_text(text)
            return json.loads(json_text)
        else:
            # Enhanced client path
            return model.generate_json(prompt)
    except Exception as e:
        if hasattr(model, '_verbose') and model._verbose:
            print(f"[Gemini] Enhanced call failed: {e}")
        raise


def _replace_ctx(ctx: SessionContext, **kwargs) -> SessionContext:
    """Return a new SessionContext with some fields replaced."""
    d = ctx.__dict__.copy()
    d.update(kwargs)
    return SessionContext(**d)


def _build_fallback(ctx: SessionContext) -> Dict:
    """
    Hardcoded fallback exercise for when all Gemini calls fail.
    Uses a basic vocabulary question in the target language.
    """
    lang_name = SUPPORTED_LANGUAGES.get(ctx.target_language, "the target language")
    fallbacks = {
        "es": [
            ("Hello", "Hola",    ["Gracias", "Adiós", "Por favor"]),
            ("Thank you", "Gracias", ["Hola", "Sí", "No"]),
            ("Goodbye", "Adiós",  ["Hola", "Bien", "Mucho"]),
        ],
        "fr": [
            ("Hello", "Bonjour", ["Merci", "Au revoir", "Oui"]),
            ("Thank you", "Merci", ["Bonjour", "Non", "Oui"]),
        ],
        "de": [
            ("Hello", "Hallo",  ["Danke", "Tschüss", "Ja"]),
        ],
    }
    options_pool = fallbacks.get(ctx.target_language, [
        ("Hello", "Hello (target)", ["Wrong1", "Wrong2", "Wrong3"]),
    ])
    native, correct, distractors = random.choice(options_pool)
    all_opts = [correct] + distractors[:3]
    random.shuffle(all_opts)
    ci = all_opts.index(correct)
    opts_fmt = [f"{chr(65+i)}) {o}" for i, o in enumerate(all_opts)]

    return dict(
        question       = f"How do you say '{native}' in {lang_name}?",
        options        = opts_fmt,
        correct_index  = ci,
        explanation    = f"'{native}' in {lang_name} is '{correct}'.",
        difficulty     = ctx.difficulty,
        hints          = ["Think about a common greeting.", "It starts with the same letter as the English word."],
        skill_tag      = "vocabulary_basic",
        exercise_type  = "multiple_choice_vocab",
        grammar_explanation = "",
        visual_breakdown    = "",
        dopamine_reward     = "🌟 Well done!",
        acceptable_answers  = [correct],
        native_word         = native,
        target_translation  = correct,
        mastery_hint        = "",
    )
