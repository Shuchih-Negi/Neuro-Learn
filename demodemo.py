"""
neurolearn_demo.py  —  NeuroLearn Adaptive Learning CLI Demo
============================================================
Demonstrates:
  1. Eye Tracking    (MediaPipe 0.10+ Tasks API, real-time webcam)
  2. ML Prediction   (rule-based, mirrors LSTM pipeline)
  3. Gemini 2.5 Flash  adaptive character-themed question generation

Usage:
  python neurolearn_demo.py --api-key YOUR_GEMINI_KEY
  python neurolearn_demo.py --api-key YOUR_GEMINI_KEY --no-camera

Requirements:
  pip install google-generativeai mediapipe opencv-python numpy

The face_landmarker.task model (~6 MB) is auto-downloaded on first run
and cached next to this script.
"""

import argparse, json, queue, threading, time, urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# ─────────────────────────────────────────────────────────────
# LANDMARK INDICES  (same 478-point mesh as old FaceMesh)
# ─────────────────────────────────────────────────────────────
LEFT_IRIS  = [474, 475, 476, 477]   # 4 pts — FaceLandmarker max index is 477
RIGHT_IRIS = [469, 470, 471, 472]   # 4 pts — index 473/478 don't exist
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_FILE = Path(__file__).parent / "face_landmarker.task"

STATE_PROFILES = {
    "Focused":     {"blink_rate":(12,25),  "pupil_dilation":(0,20),   "fixation_duration":(250,700),  "saccade_rate":(1.0,3.0),  "gaze_stability":(0.7,1.0)},
    "Drifting":    {"blink_rate":(25,45),  "pupil_dilation":(-15,5),  "fixation_duration":(50,250),   "saccade_rate":(3.0,6.0),  "gaze_stability":(0.3,0.6)},
    "Impulsive":   {"blink_rate":(30,55),  "pupil_dilation":(10,30),  "fixation_duration":(30,150),   "saccade_rate":(5.0,10.0), "gaze_stability":(0.4,0.7)},
    "Overwhelmed": {"blink_rate":(5,15),   "pupil_dilation":(20,50),  "fixation_duration":(700,2000), "saccade_rate":(0.3,1.5),  "gaze_stability":(0.6,0.9)},
}

# ─────────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────
def _ensure_model() -> Optional[str]:
    if MODEL_FILE.exists():
        return str(MODEL_FILE)
    print(f"Downloading face landmarker model (~6 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Model downloaded OK")
        return str(MODEL_FILE)
    except Exception as e:
        print(f"WARNING: Model download failed: {e}")
        print(f"  Manual: curl -L '{MODEL_URL}' -o face_landmarker.task")
        return None

# ─────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────
def _d(a, b):  return float(np.linalg.norm(np.array(a)-np.array(b)))
def _px(lm, i, w, h): return lm[i].x*w, lm[i].y*h

def _ear(lm, idx, w, h):
    c = [_px(lm,i,w,h) for i in idx]
    return (_d(c[1],c[5]) + _d(c[2],c[4])) / (2.0*_d(c[0],c[3]) + 1e-6)

def _iris_center(lm, idx, w, h):
    return float(np.mean([lm[i].x*w for i in idx])), float(np.mean([lm[i].y*h for i in idx]))

def _iris_diam(lm, idx, w, h):
    coords = np.array([_px(lm,i,w,h) for i in idx])
    center = coords.mean(0)
    # mean distance * 2 is more stable than std with only 4 landmark points
    return float(2.0 * np.mean(np.linalg.norm(coords - center, axis=1)))

# ─────────────────────────────────────────────────────────────
# EYE TRACKER  (MediaPipe Tasks API — works with mediapipe >= 0.10)
# ─────────────────────────────────────────────────────────────
class EyeTracker:
    def __init__(self, model_path: str, window_size: int = 30):
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self.landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionTaskRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        W = window_size
        self.pupil_buf  = deque(maxlen=W); self.fix_buf = deque(maxlen=100)
        self.sacc_buf   = deque(maxlen=100); self.gaze_buf = deque(maxlen=W)
        self.blink_win  = deque(maxlen=W)
        self.baseline   = None; self.last_pos = None; self.fix_start = None
        self.state = "Focused"; self.conf = 0.5
        self.frame = 0; self.blinks = 0; self.consec = 0; self.W = W
        self.EAR = 0.21

    def process_frame(self, bgr):
        import cv2, mediapipe as mp
        self.frame += 1
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r   = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        base = {"attention_state": self.state, "confidence": self.conf,
                "metrics": {}, "blink_count": self.blinks, "avg_ear": 0.0}
        if not r.face_landmarks:
            return base

        lm = r.face_landmarks[0]

        # Blink
        ear = (_ear(lm,LEFT_EYE,w,h) + _ear(lm,RIGHT_EYE,w,h)) / 2.0
        if ear < self.EAR:
            self.consec += 1; self.blink_win.append(0)
        else:
            if self.consec >= 2:
                self.blinks += 1; self.blink_win.append(1)
            else:
                self.blink_win.append(0)
            self.consec = 0

        # Pupil
        diam = (_iris_diam(lm,LEFT_IRIS,w,h) + _iris_diam(lm,RIGHT_IRIS,w,h)) / 2.0
        self.pupil_buf.append(diam)
        if self.baseline is None and len(self.pupil_buf) == self.W:
            self.baseline = float(np.mean(self.pupil_buf))

        # Gaze / saccade / fixation
        lc = _iris_center(lm,LEFT_IRIS,w,h); rc = _iris_center(lm,RIGHT_IRIS,w,h)
        gx = (lc[0]+rc[0])/2; gy = (lc[1]+rc[1])/2
        self.gaze_buf.append((gx,gy))
        if self.last_pos:
            mv = np.linalg.norm(np.array([gx,gy]) - np.array(self.last_pos))
            if mv > 30:
                self.sacc_buf.append(time.time())
                if self.fix_start:
                    self.fix_buf.append((time.time()-self.fix_start)*1000)
                    self.fix_start = None
            elif not self.fix_start:
                self.fix_start = time.time()
        self.last_pos = (gx,gy)

        # Metrics
        fps = 30; ws = len(self.blink_win)/fps
        m = {
            "blink_rate":        sum(self.blink_win)/ws*60 if ws else 0.0,
            "pupil_dilation":    ((float(np.mean(list(self.pupil_buf)[-10:])) - self.baseline)
                                   / self.baseline * 100) if self.baseline else 0.0,
            "fixation_duration": float(np.median(self.fix_buf)) if self.fix_buf else 300.0,
            "saccade_rate":      len([t for t in self.sacc_buf if time.time()-t < 2.0])/2.0,
            "gaze_stability":    max(0.0, 1.0 - float(np.mean([
                                     np.linalg.norm(np.array(g)-np.array([w/2,h/2]))
                                     for g in list(self.gaze_buf)[-10:]]))/200.0)
                                 if len(self.gaze_buf)>=10 else 1.0,
        }

        if self.frame >= self.W:
            self.state, self.conf = self._classify(m)

        return {"attention_state": self.state, "confidence": self.conf,
                "metrics": m, "blink_count": self.blinks, "avg_ear": ear}

    def _classify(self, m):
        scores = {}
        for s, prof in STATE_PROFILES.items():
            tot, n = 0.0, 0
            for k,(lo,hi) in prof.items():
                if k not in m: continue
                v = m[k]
                tot += 1.0 if lo<=v<=hi else max(0.0, 1.0-min(abs(v-lo),abs(v-hi))/((hi-lo or 1)*2))
                n += 1
            scores[s] = tot/n if n else 0.0
        best = max(scores, key=scores.get)
        return best, scores[best]

    def close(self): self.landmarker.close()

# ─────────────────────────────────────────────────────────────
# BACKGROUND THREAD
# ─────────────────────────────────────────────────────────────
class EyeTrackingThread:
    def __init__(self): self.q=queue.Queue(maxsize=1); self.running=False; self.thread=None; self.tracker=None; self.cap=None

    def start(self):
        mp = _ensure_model()
        if not mp: return False
        try:
            import cv2
            self.tracker = EyeTracker(mp, window_size=20)
            self.cap     = cv2.VideoCapture(0)
            if not self.cap.isOpened(): print("WARNING: Camera not available"); return False
            self.running = True
            self.thread  = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            print("Eye tracking started")
            return True
        except Exception as e:
            print(f"WARNING: Eye tracking init failed: {e}"); return False

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: time.sleep(0.05); continue
            result = self.tracker.process_frame(frame)
            try: self.q.get_nowait()
            except queue.Empty: pass
            self.q.put(result)
            time.sleep(0.033)

    def get_latest(self):
        try: return self.q.get_nowait()
        except queue.Empty: return None

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=2)
        if self.cap:    self.cap.release()
        if self.tracker: self.tracker.close()

# ─────────────────────────────────────────────────────────────
# ML PREDICTOR (rule-based, mirrors LSTM logic)
# ─────────────────────────────────────────────────────────────
def _rule_state(rt, correct, idle, rt_var, err_burst):
    if err_burst >= 0.67 and rt < 4.0:    return "Impulsive"
    if idle > 10 or (rt_var > 30 and not correct): return "Drifting"
    if not correct and rt > 20:            return "Overwhelmed"
    return "Focused"

class RuleBasedPredictor:
    def __init__(self): self._h = {}
    def push(self, uid, fd):
        h = self._h.setdefault(uid, [])
        h.append(fd)
        if len(h) < 3: return {"state":"Focused","confidence":0.5}
        rts = [x["rt_s"] for x in h[-5:]]
        state = _rule_state(fd["rt_s"], fd["correct"], fd.get("idle_s",0),
                            float(np.var(rts)), sum(1 for x in h[-3:] if not x["correct"])/3)
        return {"state": state, "confidence": 0.85}

# ─────────────────────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────────────────────
class GeminiQuestionGenerator:
    def __init__(self, api_key):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model   = genai.GenerativeModel("gemini-2.5-flash")
        self.history: List[str] = []

    def generate(self, character, topic, state, difficulty):
        instr = {
            "Focused":     "a challenging multi-step problem",
            "Drifting":    "a short playful problem with a fun twist to re-engage the student",
            "Impulsive":   "a tricky problem that starts with 'Read carefully:' to slow the student down",
            "Overwhelmed": "ONLY the very first small step — keep it simple and encouraging",
        }.get(state, "a standard problem")

        prev = ("\n\nALREADY ASKED (do not repeat):\n" +
                "\n".join(f"- {q}" for q in self.history[-3:])) if self.history else ""

        prompt = f"""You are a friendly math tutor for children (grades 3-5).
Generate ONE math question about {topic} for a student who loves {character}.

ATTENTION STATE: {state}
DIFFICULTY: {difficulty}/5
TASK: Create {instr}.

Rules:
- Wrap the problem in a short story starring {character}
- Keep it age-appropriate and fun
- Exactly 4 multiple-choice options labelled A, B, C, D
- Only one option is correct
{prev}

Return ONLY valid JSON, no markdown fences:
{{"question":"...","options":["A) ...","B) ...","C) ...","D) ..."],"correct_index":0,"explanation":"...","difficulty":{difficulty}}}"""

        try:
            text = self.model.generate_content(prompt).text.strip()
            if "```" in text:
                for part in text.split("```"):
                    s = part.lstrip("json").strip()
                    if s.startswith("{"): text = s; break
            data = json.loads(text)
            if not all(k in data for k in ["question","options","correct_index","explanation"]):
                raise ValueError("Missing fields")
            if len(data["options"]) != 4: raise ValueError("Need 4 options")
            self.history.append(data["question"])
            return data
        except Exception as e:
            print(f"\nWARNING: Gemini error ({e}) - using fallback")
            return self._fallback(character, topic)

    def _fallback(self, character, topic):
        fb = {
            "addition":       {"question":f"{character} collected 7 gems in level 1 and 5 in level 2. How many total?","options":["A) 10","B) 11","C) 12","D) 13"],"correct_index":2,"explanation":"7+5=12","difficulty":1},
            "subtraction":    {"question":f"{character} had 15 coins and spent 6. How many left?","options":["A) 7","B) 8","C) 9","D) 10"],"correct_index":2,"explanation":"15-6=9","difficulty":1},
            "multiplication": {"question":f"{character} has 4 bags with 6 apples each. Total?","options":["A) 20","B) 22","C) 24","D) 26"],"correct_index":2,"explanation":"4x6=24","difficulty":2},
            "fractions":      {"question":f"{character} ate 2/8 of a pizza. Simplified?","options":["A) 1/2","B) 1/4","C) 1/3","D) 2/4"],"correct_index":1,"explanation":"2/8=1/4","difficulty":3},
        }
        d = fb.get(topic.lower(), fb["addition"])
        self.history.append(d["question"]); return d

# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────
C = {"Focused":"\033[92m","Drifting":"\033[93m","Impulsive":"\033[91m","Overwhelmed":"\033[95m"}
R = "\033[0m"; B = "\033[1m"

def state_display(state, conf, metrics):
    c = C.get(state,"")
    print(f"\n{c}{'─'*62}")
    print(f"  ML STATE: {state.upper():12s}  Confidence: {conf:.0%}")
    if metrics:
        print(f"  Blink: {metrics.get('blink_rate',0):4.0f}/min  "
              f"Pupil: {metrics.get('pupil_dilation',0):+5.1f}%  "
              f"Fixation: {metrics.get('fixation_duration',0):4.0f}ms  "
              f"Saccade: {metrics.get('saccade_rate',0):.1f}/s")
    print(f"{'─'*62}{R}")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key",   required=True)
    ap.add_argument("--no-camera", action="store_true")
    args = ap.parse_args()

    print(f"\n{B}{'='*62}\n   NeuroLearn - Adaptive Learning Demo\n{'='*62}{R}")
    print(f"\n{B}Initializing...{R}")

    gemini    = GeminiQuestionGenerator(args.api_key)
    predictor = RuleBasedPredictor()
    print("Gemini 2.5 Flash ready")
    print("Rule-based ML predictor ready (mirrors LSTM pipeline)")

    eye: Optional[EyeTrackingThread] = None
    if not args.no_camera:
        eye = EyeTrackingThread()
        if not eye.start(): eye = None
        else: time.sleep(1.5)

    print("\n" + "─"*62)
    character = input("Your learning buddy (e.g. Spiderman, Elsa, Pikachu): ").strip() or "Dragon"
    topic     = input("Math topic (addition/subtraction/multiplication/fractions): ").strip().lower() or "addition"
    print("─"*62)
    print(f"\nStarting 10-question session | topic: {B}{topic}{R} | buddy: {B}{character}{R}")
    print(f"Eye Tracking: {'ENABLED' if eye else 'DISABLED (--no-camera or model missing)'}")
    input("\nPress ENTER when ready... ")

    stats = {"correct":0,"total":0,"states":[],"times":[]}
    current_state = "Focused"; difficulty = 2

    for q_num in range(1, 11):
        print(f"\n\n{B}{'='*62}\n   QUESTION {q_num}/10   |   Difficulty {difficulty}/5\n{'='*62}{R}")

        eye_result = eye_metrics = {}
        if eye:
            eye_result = eye.get_latest() or {}
            if eye_result.get("metrics"):
                eye_metrics = eye_result["metrics"]
                es = eye_result["attention_state"]
                ec = eye_result.get("confidence",0)
                print(f"Eye tracker  ->  {C.get(es,'')}{es}{R}  (conf {ec:.0%}, blinks: {eye_result.get('blink_count',0)})")

        print(f"Gemini generating {C.get(current_state,'')}{current_state}{R} question ...")
        q = gemini.generate(character, topic, current_state, difficulty)

        print(f"\n{B}{q['question']}{R}\n")
        for opt in q["options"]: print(f"  {opt}")

        t0 = time.time()
        while True:
            ans = input("\nYour answer (A/B/C/D): ").strip().upper()
            if ans in "ABCD" and len(ans)==1: break
            print("  Please type A, B, C, or D")
        rt = time.time() - t0

        correct     = ord(ans)-ord("A") == q["correct_index"]
        correct_ltr = chr(ord("A")+q["correct_index"])

        if correct:
            print(f"\n  {C['Focused']}Correct!{R} {q['explanation']}")
            stats["correct"] += 1
        else:
            print(f"\n  Wrong - answer was {correct_ltr}. {q['explanation']}")

        stats["total"] += 1; stats["times"].append(rt)

        ml = predictor.push(1, {"rt_s":rt,"correct":int(correct),"idle_s":max(0,rt-5),
                                "rt_mean":sum(stats["times"])/len(stats["times"])})

        # Merge ML + eye state
        if eye_result and eye_result.get("confidence",0) > 0.55:
            current_state = eye_result["attention_state"]
        else:
            current_state = ml["state"]

        stats["states"].append(current_state)
        state_display(current_state, ml["confidence"], eye_metrics if isinstance(eye_metrics,dict) else {})

        if current_state=="Focused" and correct:
            difficulty = min(5,difficulty+1); print("  Increasing difficulty...")
        elif current_state=="Overwhelmed":
            difficulty = max(1,difficulty-1); print("  Simplifying next question...")
        elif current_state=="Impulsive":   print("  Next question will say 'Read carefully:'")
        elif current_state=="Drifting":    print("  Next question will be more engaging!")

        time.sleep(1.2)

    # Summary
    acc = stats["correct"]/stats["total"]
    avg = sum(stats["times"])/len(stats["times"])
    print(f"\n\n{B}{'='*62}\n   SESSION COMPLETE!\n{'='*62}{R}")
    print(f"\nScore: {stats['correct']}/{stats['total']} ({acc:.0%})   Avg time: {avg:.1f}s")

    counts = Counter(stats["states"])
    print("\nAttention breakdown:")
    for s,n in counts.most_common():
        print(f"  {C.get(s,'')}{s:12s}{R} {'|'*n} ({n})")

    print("\n" + {
        "Focused":     "Great focus! Try harder topics next time.",
        "Drifting":    "Attention wandered - try shorter breaks.",
        "Impulsive":   "Several rushed answers - slow down next time!",
        "Overwhelmed": "Difficulty was too high - we'll start easier.",
    }[counts.most_common(1)[0][0]])

    if eye: eye.stop()
    print(f"\n{B}{'='*62}\n   Thanks for using NeuroLearn!\n{'='*62}{R}\n")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nSession interrupted")
    except Exception as e:
        import traceback; print(f"\nError: {e}"); traceback.print_exc()