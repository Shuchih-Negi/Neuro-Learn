"""
ml/nlp_model.py  — NeuroLearn Language Edition
================================================
NLP engine for grammar correction, error classification, and
flexible (semantic) answer validation.

Research basis
──────────────
• Levenshtein (1966) — Edit Distance:
    Used as a fast pre-filter before semantic similarity scoring.
    If edit distance is low enough, accept as correct.

• Devlin et al. (2019) — BERT / Transformers:
    Sentence embeddings capture meaning independently of surface form.
    "I eat apples" ≈ "Apples are what I eat" in embedding space.
    Used for semantic similarity scoring of learner answers.

• Swain (1985) — Output Hypothesis:
    Learner production errors are diagnostic. Our error classifier
    (vocab / grammar / word_order / spelling) feeds back into the
    LSTM mastery model as the error_type feature.

• Lyster & Ranta (1997) — Corrective Feedback Types:
    Recasting (showing correct form without explicit rejection) is most
    effective for language acquisition. This module produces recast messages.

Design
──────
Two operational modes:
    1. Strict mode   — exact match or Levenshtein close enough (< threshold)
    2. Semantic mode — sentence transformer embedding cosine similarity

The semantic mode is used when the acceptable_answers list provided by
the moderator indicates that semantic variants should be accepted.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional heavy dependencies — gracefully degraded if not installed
_transformers_ok = False
_sentence_transformer = None

try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformer = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    _transformers_ok = True
    print("[NLPModel] Sentence transformer loaded.")
except ImportError:
    print("[NLPModel] sentence-transformers not installed; using edit-distance fallback.")
    print("           pip install sentence-transformers  (optional, improves validation)")


# ─────────────────────────────────────────────────────────────
# Error type taxonomy
# ─────────────────────────────────────────────────────────────

ERROR_TYPES = {
    "correct":       "No error",
    "vocabulary":    "Wrong word chosen (vocab gap)",
    "grammar":       "Correct word, wrong form (conjugation / agreement)",
    "word_order":    "Words correct but wrong order (syntax)",
    "spelling":      "Phonetically close but wrong spelling",
    "missing_accent":"Correct word but missing accent mark",
    "anglicism":     "English word inserted into target language",
}


# ─────────────────────────────────────────────────────────────
# Text normalisation
# ─────────────────────────────────────────────────────────────

def _normalise(text: str, strip_accents: bool = False) -> str:
    """Lowercase, strip punctuation, optionally remove diacritics."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    if strip_accents:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    return text.strip()


# ─────────────────────────────────────────────────────────────
# Edit distance (Levenshtein)
# ─────────────────────────────────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    """Classic DP Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,  # delete
                curr[j] + 1,      # insert
                prev[j] + (0 if c1 == c2 else 1),  # replace
            ))
        prev = curr
    return prev[-1]


def _normalised_edit_distance(s1: str, s2: str) -> float:
    """Edit distance normalised to [0, 1] by max string length."""
    dist = levenshtein(s1, s2)
    max_len = max(len(s1), len(s2), 1)
    return dist / max_len


# ─────────────────────────────────────────────────────────────
# Semantic similarity (sentence transformers)
# ─────────────────────────────────────────────────────────────

def _semantic_similarity(s1: str, s2: str) -> float:
    """Cosine similarity of multilingual sentence embeddings. Returns 0-1."""
    if not _transformers_ok or _sentence_transformer is None:
        return 0.0
    try:
        embs = _sentence_transformer.encode([s1, s2], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# Error classifier
# ─────────────────────────────────────────────────────────────

def classify_error(
    learner_answer: str,
    correct_answer: str,
    acceptable_answers: List[str],
    target_language: str = "es",
) -> str:
    """
    Classify the type of error in a learner's answer.
    Returns one of the ERROR_TYPES keys.

    Heuristics (lightweight, no external model required):
    1. If semantically close but different surface → vocabulary or grammar
    2. Same words, different order → word_order
    3. Very close edit distance → spelling or missing_accent
    4. English-origin word detected → anglicism
    """
    la_norm = _normalise(learner_answer)
    ca_norm = _normalise(correct_answer)

    if la_norm == ca_norm:
        return "correct"

    # Check acceptable variants
    for acc in acceptable_answers:
        if _normalise(acc) == la_norm:
            return "correct"

    ed = _normalised_edit_distance(la_norm, ca_norm)

    # Accent stripping — same word without accent?
    if _normalise(la_norm, strip_accents=True) == _normalise(ca_norm, strip_accents=True):
        return "missing_accent"

    # Word set overlap for word-order detection
    la_words = set(la_norm.split())
    ca_words = set(ca_norm.split())
    if la_words == ca_words and la_norm != ca_norm:
        return "word_order"

    # Very close edit distance → spelling
    if ed <= 0.25:
        return "spelling"

    # English word check: if learner wrote an English word not in target vocab
    english_words = {"the", "is", "are", "i", "you", "he", "she", "we", "they",
                     "to", "and", "or", "not", "no", "yes", "of", "in", "on", "at"}
    la_word_set = set(la_norm.split())
    if la_word_set & english_words and target_language != "en":
        return "anglicism"

    # Semantic similarity — if meaning is close, it's grammar/vocab not wild guessing
    if _transformers_ok:
        sim = _semantic_similarity(learner_answer, correct_answer)
        if sim > 0.75:
            return "grammar"

    return "vocabulary"


# ─────────────────────────────────────────────────────────────
# Answer validator
# ─────────────────────────────────────────────────────────────

class AnswerValidator:
    """
    Validates learner answers using a 3-tier system:
        1. Exact match (normalised)         → correct
        2. Acceptable variants match        → correct
        3. Accent-stripped match            → correct (partial credit flagged)
        4. Edit distance ≤ threshold        → close (configurable)
        5. Semantic similarity ≥ threshold  → semantically correct

    Research: Swain (1985) — output-hypothesis: credit meaningful output
    even if surface form is imperfect, to encourage production.
    """

    def __init__(
        self,
        edit_distance_threshold: float = 0.20,
        semantic_threshold:      float = 0.82,
    ):
        self.edit_threshold = edit_distance_threshold
        self.sem_threshold  = semantic_threshold

    def validate(
        self,
        learner_answer:    str,
        correct_answer:    str,
        acceptable_answers: List[str],
        target_language:   str = "es",
    ) -> Dict:
        """
        Returns:
            {
              "is_correct": bool,
              "is_partial": bool,       # semantically right but surface wrong
              "error_type": str,        # from ERROR_TYPES
              "score": float,           # 0.0–1.0
              "recast": str,            # Lyster & Ranta recast message
              "method": str,            # exact / variant / accent / edit / semantic / wrong
            }
        """
        la = learner_answer.strip()
        ca = correct_answer.strip()
        la_norm = _normalise(la)
        ca_norm = _normalise(ca)
        all_correct = [ca] + list(acceptable_answers)

        # ── Tier 1: exact match ──────────────────────────────
        if la_norm == ca_norm:
            return self._result(True, False, "correct", 1.0,
                                f"✅ Perfect! '{ca}' is correct.", "exact")

        # ── Tier 2: acceptable variants ──────────────────────
        for variant in acceptable_answers:
            if la_norm == _normalise(variant):
                return self._result(True, False, "correct", 1.0,
                                    f"✅ Great! '{variant}' is also accepted.", "variant")

        # ── Tier 3: accent-stripped match ────────────────────
        la_bare = _normalise(la, strip_accents=True)
        for acc_ans in all_correct:
            if la_bare == _normalise(acc_ans, strip_accents=True):
                return self._result(
                    True, True, "missing_accent", 0.85,
                    f"Almost perfect! Remember the accent: '{ca}'",
                    "accent"
                )

        # ── Tier 4: edit distance ────────────────────────────
        best_ed = min(_normalised_edit_distance(la_norm, _normalise(a)) for a in all_correct)
        if best_ed <= self.edit_threshold:
            error = classify_error(la, ca, acceptable_answers, target_language)
            return self._result(
                True, True, error, 0.75,
                f"Very close! The correct form is '{ca}'",
                "edit"
            )

        # ── Tier 5: semantic similarity ───────────────────────
        if _transformers_ok:
            best_sim = max(
                _semantic_similarity(la, a) for a in all_correct
            )
            if best_sim >= self.sem_threshold:
                error = classify_error(la, ca, acceptable_answers, target_language)
                return self._result(
                    True, True, error, 0.70,
                    f"Good meaning! In {target_language} we say: '{ca}'",
                    "semantic"
                )

        # ── Tier 6: wrong ─────────────────────────────────────
        error = classify_error(la, ca, acceptable_answers, target_language)
        return self._result(
            False, False, error, 0.0,
            f"In {target_language}, this is '{ca}'. Try again!",
            "wrong"
        )

    @staticmethod
    def _result(
        is_correct: bool, is_partial: bool, error_type: str,
        score: float, recast: str, method: str,
    ) -> Dict:
        return {
            "is_correct":  is_correct,
            "is_partial":  is_partial,
            "error_type":  error_type,
            "score":       score,
            "recast":      recast,
            "method":      method,
        }


# ─────────────────────────────────────────────────────────────
# Grammar explainer (simple rule-based for common patterns)
# ─────────────────────────────────────────────────────────────

GRAMMAR_PATTERNS = {
    "es": {
        "ar_verb_yo": {
            "pattern": r"\w+ar$",
            "rule": "For -AR verbs: yo → -o (hablo, como, escribo)",
            "visual": "🟦 yo + verb_stem + 🟨 -o",
        },
        "gender_noun": {
            "pattern": r"(el|la)\s+\w+",
            "rule": "Nouns ending in -o are usually masculine (el), -a feminine (la)",
            "visual": "🔵 el + masculine noun  |  🔴 la + feminine noun",
        },
    },
    "fr": {
        "er_verb_je": {
            "pattern": r"\w+er$",
            "rule": "For -ER verbs: je → -e (je parle, je mange)",
            "visual": "🟦 je + verb_stem + 🟨 -e",
        },
    },
}


def get_grammar_hint(text: str, language: str = "es") -> Optional[str]:
    """
    Return a grammar visual hint if a known pattern is detected.
    Lightweight — no ML required.
    """
    patterns = GRAMMAR_PATTERNS.get(language, {})
    for name, info in patterns.items():
        if re.search(info["pattern"], text, re.IGNORECASE):
            return f"{info['rule']}\n{info['visual']}"
    return None


# ─────────────────────────────────────────────────────────────
# Public API object
# ─────────────────────────────────────────────────────────────

class NLPModel:
    """
    Main NLP engine. Used by app.py to validate answers and
    generate recast feedback.

    Usage
    -----
        nlp = NLPModel()
        result = nlp.validate_answer(
            learner_answer="voy",
            correct_answer="voy",
            acceptable_answers=["me voy", "yo voy"],
            target_language="es",
        )
        print(result["recast"])   # "✅ Perfect! 'voy' is correct."
    """

    def __init__(self):
        self.validator = AnswerValidator()

    def validate_answer(
        self,
        learner_answer:     str,
        correct_answer:     str,
        acceptable_answers: List[str],
        target_language:    str = "es",
    ) -> Dict:
        return self.validator.validate(
            learner_answer, correct_answer, acceptable_answers, target_language
        )

    def classify_error(
        self,
        learner_answer:     str,
        correct_answer:     str,
        acceptable_answers: List[str],
        target_language:    str = "es",
    ) -> str:
        return classify_error(
            learner_answer, correct_answer, acceptable_answers, target_language
        )

    def get_grammar_hint(self, text: str, language: str = "es") -> Optional[str]:
        return get_grammar_hint(text, language)
