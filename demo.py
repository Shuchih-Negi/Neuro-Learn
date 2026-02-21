"""
demo_cli.py
===========
Complete interactive CLI demo that shows:
  1. Eye tracking (webcam, real-time)
  2. ML model (LSTM or rule-based)
  3. Gemini question generation (adaptive based on state)

Flow:
  - User picks character + math topic
  - Loop 10 questions:
      → Eye tracker detects state
      → ML model predicts attention state
      → Gemini generates adaptive question
      → User answers
      → System logs metrics and adapts

Usage:
  python demo_cli.py --api-key YOUR_GEMINI_KEY
  python demo_cli.py --api-key YOUR_GEMINI_KEY --no-camera  # skip eye tracking

Requirements:
  pip install google-generativeai mediapipe opencv-python --break-system-packages
"""

import argparse
import time
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import google.generativeai as genai
from typing import Dict, Optional, List
import threading
import queue


# ─────────────────────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────────────────────

class GeminiQuestionGenerator:
    """Generates adaptive math questions using Gemini 2.0 Flash."""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.history = []  # track what questions were asked
    
    def generate_question(
        self,
        character: str,
        topic: str,
        state: str,
        difficulty_level: int,
        previous_questions: List[str] = None,
    ) -> Dict:
        """
        Generate one math question adapted to the current attention state.
        
        Parameters
        ----------
        character : str
            User's chosen character (e.g. "Spiderman", "Elsa", "Pikachu")
        topic : str
            Math topic (e.g. "addition", "fractions", "multiplication")
        state : str
            Current attention state (Focused/Drifting/Impulsive/Overwhelmed)
        difficulty_level : int
            1 (easy) to 5 (hard)
        previous_questions : list
            To avoid repetition
        
        Returns
        -------
        dict with:
          - question: str
          - options: list[str] (4 multiple choice)
          - correct_index: int (0-3)
          - explanation: str
          - difficulty: int
        """
        # Build prompt based on state
        state_instructions = {
            "Focused": "challenging multi-step problem",
            "Drifting": "short, engaging problem with a fun twist",
            "Impulsive": "problem that requires careful reading with a 'read carefully' warning",
            "Overwhelmed": "break a complex problem into the FIRST SMALL STEP only",
        }
        
        instruction = state_instructions.get(state, "standard problem")
        
        # Build previous questions context
        prev_context = ""
        if previous_questions:
            prev_context = f"\n\nALREADY ASKED (don't repeat):\n" + "\n".join(f"- {q}" for q in previous_questions[-3:])
        
        prompt = f"""You are a math tutor for children. Generate ONE math question about {topic} for a student who loves {character}.

ATTENTION STATE: {state}
DIFFICULTY: {difficulty_level}/5
INSTRUCTION: Create a {instruction}.

Requirements:
- Frame the problem as a short story involving {character}
- Make it age-appropriate (grades 3-5)
- For Overwhelmed state: ONLY ask the first step of a multi-step problem
- For Impulsive state: Add "Read carefully:" at the start
- Provide 4 multiple choice options (A, B, C, D)
- Mark the correct answer

{prev_context}

Return ONLY valid JSON in this exact format:
{{
  "question": "Story-based math question here",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct_index": 0,
  "explanation": "Brief explanation why this answer is correct",
  "difficulty": {difficulty_level}
}}"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Extract JSON (handle markdown fences)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(text)
            
            # Validate structure
            required = ["question", "options", "correct_index", "explanation"]
            if not all(k in result for k in required):
                raise ValueError("Missing required fields in Gemini response")
            
            if len(result["options"]) != 4:
                raise ValueError("Must have exactly 4 options")
            
            self.history.append(result["question"])
            return result
            
        except Exception as e:
            print(f"\n⚠️  Gemini generation failed: {e}")
            print("Using fallback question...")
            return self._fallback_question(character, topic, state)
    
    def _fallback_question(self, character: str, topic: str, state: str) -> Dict:
        """Hardcoded fallback questions when Gemini fails."""
        fallbacks = {
            "addition": {
                "question": f"{character} collected 7 stars in level 1 and 5 stars in level 2. How many stars total?",
                "options": ["A) 10", "B) 11", "C) 12", "D) 13"],
                "correct_index": 2,
                "explanation": "7 + 5 = 12",
                "difficulty": 1,
            },
            "multiplication": {
                "question": f"{character} has 4 boxes with 6 items each. How many items in total?",
                "options": ["A) 20", "B) 22", "C) 24", "D) 26"],
                "correct_index": 2,
                "explanation": "4 × 6 = 24",
                "difficulty": 2,
            },
            "fractions": {
                "question": f"{character} ate 2/8 of a pizza. What is this simplified?",
                "options": ["A) 1/2", "B) 1/4", "C) 1/3", "D) 2/4"],
                "correct_index": 1,
                "explanation": "2/8 = 1/4 (divide both by 2)",
                "difficulty": 3,
            },
        }
        
        return fallbacks.get(topic.lower(), fallbacks["addition"])


# ─────────────────────────────────────────────────────────────
# EYE TRACKING THREAD (non-blocking)
# ─────────────────────────────────────────────────────────────

class EyeTrackingThread:
    """Runs eye tracker in background thread, puts results in queue."""
    
    def __init__(self):
        self.queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.tracker = None
    
    def start(self):
        """Start background eye tracking."""
        try:
            from eye_tracker import EyeTracker
            import cv2
            
            self.tracker = EyeTracker(window_size=20)  # shorter window for demo
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("⚠️  Camera not available, eye tracking disabled")
                return False
            
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            print("✓ Eye tracking started")
            return True
            
        except Exception as e:
            print(f"⚠️  Eye tracking initialization failed: {e}")
            return False
    
    def _run(self):
        """Background loop that processes frames."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            result = self.tracker.process_frame(frame)
            
            # Put latest result in queue (discard old if queue full)
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            
            self.queue.put(result)
            time.sleep(0.03)  # ~30 FPS
    
    def get_latest(self) -> Optional[Dict]:
        """Get most recent eye tracking result (non-blocking)."""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop background thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if hasattr(self, 'cap'):
            self.cap.release()


# ─────────────────────────────────────────────────────────────
# ML MODEL INTEGRATION
# ─────────────────────────────────────────────────────────────

def get_ml_predictor(use_lstm: bool = False):
    """Load ML model or use rule-based fallback."""
    if use_lstm:
        try:
            from models.lstm_model import AttentionPredictor
            predictor = AttentionPredictor("models/best_lstm.pt", "models/scaler.pkl")
            print("✓ LSTM model loaded")
            return predictor, "LSTM"
        except Exception as e:
            print(f"⚠️  LSTM not available ({e}), using rule-based")
    
    # Rule-based fallback
    from utils.feature_engineering import rule_based_state, FEATURE_COLS
    
    class RuleBasedPredictor:
        def __init__(self):
            self._buffers = {}
        
        def push(self, user_id: int, feature_dict: dict):
            history = self._buffers.setdefault(user_id, [])
            history.append(feature_dict)
            
            if len(history) < 3:
                return {"state": "Focused", "confidence": 0.5}
            
            recent = history[-5:]
            rts = [h["rt_s"] for h in recent]
            rt_var = sum((x - sum(rts)/len(rts))**2 for x in rts) / len(rts)
            error_burst = sum(1 for h in recent[-3:] if not h["correct"]) / 3.0
            
            state = rule_based_state(
                rt_s=feature_dict["rt_s"],
                correct=feature_dict["correct"],
                attempt_count=feature_dict["attempt_count"],
                hint_count=feature_dict["hint_count"],
                idle_s=feature_dict["idle_s"],
                rt_variance=rt_var,
                error_burst=error_burst,
            )
            
            return {"state": state, "confidence": 0.85}
    
    print("✓ Rule-based classifier loaded")
    return RuleBasedPredictor(), "Rule-based"


# ─────────────────────────────────────────────────────────────
# MAIN CLI DEMO
# ─────────────────────────────────────────────────────────────

def print_banner():
    print("\n" + "="*60)
    print("   🐉 BUDDY — Adaptive Math Learning Demo")
    print("="*60)


def print_state_banner(state: str, confidence: float, eye_metrics: Dict = None):
    """Print current attention state with color."""
    colors = {
        "Focused":     "\033[92m",  # green
        "Drifting":    "\033[93m",  # yellow
        "Impulsive":   "\033[91m",  # red
        "Overwhelmed": "\033[95m",  # magenta
    }
    reset = "\033[0m"
    
    color = colors.get(state, "")
    print(f"\n{color}{'─'*60}")
    print(f"  STATE: {state.upper()} | Confidence: {confidence:.0%}")
    
    if eye_metrics:
        print(f"  👁️  Blink: {eye_metrics.get('blink_rate', 0):.0f}/min | "
              f"Pupil: {eye_metrics.get('pupil_dilation', 0):+.0f}% | "
              f"Fixation: {eye_metrics.get('fixation_duration', 0):.0f}ms")
    
    print(f"{'─'*60}{reset}\n")


def main():
    parser = argparse.ArgumentParser(description="BUDDY Interactive Demo")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--no-camera", action="store_true", help="Skip eye tracking")
    parser.add_argument("--use-lstm", action="store_true", help="Try loading LSTM model")
    args = parser.parse_args()
    
    print_banner()
    
    # ── Initialize components ────────────────────────────────
    
    print("\n🔧 Initializing components...")
    
    gemini = GeminiQuestionGenerator(args.api_key)
    print("✓ Gemini API connected")
    
    ml_predictor, ml_type = get_ml_predictor(use_lstm=args.use_lstm)
    
    eye_tracker = None
    if not args.no_camera:
        eye_tracker = EyeTrackingThread()
        eye_tracker.start()
        time.sleep(1.0)  # let camera warm up
    
    # ── Setup session ────────────────────────────────────────
    
    print("\n" + "─"*60)
    character = input("🎭 Choose your learning buddy (e.g. Spiderman, Elsa, Pikachu): ").strip() or "Dragon"
    topic     = input("📚 Choose math topic (addition/multiplication/fractions): ").strip().lower() or "addition"
    print("─"*60)
    
    print(f"\n🎯 Starting 10-question session on {topic} with {character}!")
    print(f"🧠 ML Engine: {ml_type}")
    print(f"👁️  Eye Tracking: {'Enabled' if eye_tracker else 'Disabled'}")
    
    input("\nPress ENTER when ready...")
    
    # ── Question loop ────────────────────────────────────────
    
    session_stats = {
        "correct": 0,
        "total": 0,
        "states": [],
        "response_times": [],
    }
    
    current_state = "Focused"
    difficulty = 2  # start at medium
    
    for q_num in range(1, 11):
        print(f"\n\n{'='*60}")
        print(f"  QUESTION {q_num}/10")
        print(f"{'='*60}")
        
        # Get eye tracking data if available
        eye_result = None
        eye_metrics = {}
        if eye_tracker:
            eye_result = eye_tracker.get_latest()
            if eye_result and eye_result.get("metrics"):
                eye_metrics = eye_result["metrics"]
                # Eye tracker also predicts state
                eye_state = eye_result["attention_state"]
                print(f"👁️  Eye tracker detects: {eye_state}")
        
        # Generate question adapted to current state
        print(f"🤖 Generating {current_state.lower()} question (difficulty {difficulty})...")
        
        question_data = gemini.generate_question(
            character=character,
            topic=topic,
            state=current_state,
            difficulty_level=difficulty,
            previous_questions=gemini.history[-3:],
        )
        
        # Display question
        print(f"\n📝 {question_data['question']}")
        print()
        for option in question_data["options"]:
            print(f"   {option}")
        
        # Start timer
        start_time = time.time()
        
        # Get answer
        while True:
            user_input = input("\n👉 Your answer (A/B/C/D): ").strip().upper()
            if user_input in ["A", "B", "C", "D"]:
                break
            print("⚠️  Please enter A, B, C, or D")
        
        # Stop timer
        response_time = time.time() - start_time
        
        # Check answer
        user_idx = ord(user_input) - ord("A")
        correct = (user_idx == question_data["correct_index"])
        
        correct_letter = chr(ord("A") + question_data["correct_index"])
        
        if correct:
            print(f"\n✅ Correct! {question_data['explanation']}")
            session_stats["correct"] += 1
        else:
            print(f"\n❌ Wrong. The answer was {correct_letter}. {question_data['explanation']}")
        
        session_stats["total"] += 1
        session_stats["response_times"].append(response_time)
        
        # ── ML prediction ────────────────────────────────────
        
        # Build feature dict for ML model
        features = {
            "rt_s": response_time,
            "correct": int(correct),
            "attempt_count": 1,
            "hint_count": 0,
            "hint_ratio": 0.0,
            "idle_s": max(0, response_time - 2.0),
            "rt_mean": sum(session_stats["response_times"]) / len(session_stats["response_times"]),
            "rt_variance": 0.0,  # would compute from history
            "rt_trend": 0.0,
            "error_rate": 1 - (session_stats["correct"] / session_stats["total"]),
            "error_burst": 0.0,
            "attempt_mean": 1.0,
            "hint_rate": 0.0,
        }
        
        # Get ML prediction
        ml_result = ml_predictor.push(user_id=1, feature_dict=features)
        current_state = ml_result["state"]
        confidence = ml_result.get("confidence", 0.8)
        
        session_stats["states"].append(current_state)
        
        # Display state
        print_state_banner(current_state, confidence, eye_metrics)
        
        # Adapt difficulty for next question
        if current_state == "Focused" and correct:
            difficulty = min(5, difficulty + 1)
            print("📈 Increasing difficulty...")
        elif current_state == "Overwhelmed":
            difficulty = max(1, difficulty - 1)
            print("📉 Simplifying next question...")
        elif current_state == "Impulsive":
            print("⚠️  Next question will require careful reading...")
        elif current_state == "Drifting":
            print("🎯 Next question will be more engaging...")
        
        # Brief pause
        time.sleep(1.5)
    
    # ── Session summary ──────────────────────────────────────
    
    print("\n\n" + "="*60)
    print("   🎉 SESSION COMPLETE!")
    print("="*60)
    
    accuracy = session_stats["correct"] / session_stats["total"]
    avg_rt = sum(session_stats["response_times"]) / len(session_stats["response_times"])
    
    print(f"\n📊 Results:")
    print(f"   Accuracy:  {accuracy:.0%} ({session_stats['correct']}/{session_stats['total']})")
    print(f"   Avg Time:  {avg_rt:.1f}s")
    print(f"   ML Engine: {ml_type}")
    
    print(f"\n🧠 Attention States This Session:")
    from collections import Counter
    state_counts = Counter(session_stats["states"])
    for state, count in state_counts.most_common():
        pct = count / len(session_stats["states"])
        print(f"   {state:12s} {count:2d} questions ({pct:.0%})")
    
    dominant = state_counts.most_common(1)[0][0]
    
    insights = {
        "Focused": "🎯 Great focus throughout! Try harder topics next time.",
        "Drifting": "💤 Your attention wandered mid-session. Try shorter breaks between questions.",
        "Impulsive": "⚡ You rushed several answers. Next time, use the 'slow mode'.",
        "Overwhelmed": "😰 The difficulty was too high. We'll start easier next session.",
    }
    
    print(f"\n💡 Insight: {insights[dominant]}")
    
    # Cleanup
    if eye_tracker:
        eye_tracker.stop()
    
    print("\n" + "="*60)
    print("   Thanks for using BUDDY! 🐉")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()