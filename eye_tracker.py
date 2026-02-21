"""
eye_tracker.py
==============
Real-time eye tracking module that detects attention states from webcam video.
Maps eye metrics → Focused | Drifting | Impulsive | Overwhelmed

Uses MediaPipe Face Mesh (no specialized hardware needed, just a webcam).

Research foundation:
  - Blink rate changes with cognitive load (Stern et al., 1994)
  - Pupil dilation correlates with mental effort (Beatty, 1982; Kahneman, 1973)
  - Fixation duration increases with task difficulty (Hyönä et al., 1995)
  - Saccade rate differentiates attention states (Di Stasi et al., 2018)

Installation:
  pip install mediapipe opencv-python numpy --break-system-packages

Usage:
  python eye_tracker.py                    # Live webcam demo
  python eye_tracker.py --video test.mp4   # Process video file
  python eye_tracker.py --integrate        # Save to dataset format
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from collections import deque
from typing import Optional, Tuple, Dict, List
import json


# ─────────────────────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

# MediaPipe iris landmarks (5 per eye, total 10)
LEFT_IRIS  = [474, 475, 476, 477, 478]   # indices in 478-point mesh
RIGHT_IRIS = [469, 470, 471, 472, 473]

# Eye contour landmarks (for blink detection via Eye Aspect Ratio)
# Based on Soukupová & Čech (2016): Real-Time Eye Blink Detection using Facial Landmarks
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


# ─────────────────────────────────────────────────────────────
# ATTENTION STATE THRESHOLDS (research-backed + empirically tuned)
# ─────────────────────────────────────────────────────────────

# Multi-metric classification rules
# Each state has a characteristic "signature" across multiple metrics

STATE_PROFILES = {
    "Focused": {
        "blink_rate":        (12, 25),   # blinks/min - normal engaged rate
        "pupil_dilation":    (0,  20),   # % change from baseline - slight arousal
        "fixation_duration": (250, 700), # ms - stable fixations
        "saccade_rate":      (1.0, 3.0), # per second - controlled movements
        "gaze_stability":    (0.7, 1.0), # ratio - eyes stay centered
    },
    "Drifting": {
        "blink_rate":        (25, 45),   # increased when disengaged
        "pupil_dilation":    (-15, 5),   # constricted, low arousal
        "fixation_duration": (50,  250), # short, scattered fixations
        "saccade_rate":      (3.0, 6.0), # wandering gaze
        "gaze_stability":    (0.3, 0.6), # eyes moving off-center
    },
    "Impulsive": {
        "blink_rate":        (30, 55),   # rapid, restless blinking
        "pupil_dilation":    (10, 30),   # moderate arousal
        "fixation_duration": (30,  150), # very brief fixations, scanning
        "saccade_rate":      (5.0, 10.0),# hyperactive eye movements
        "gaze_stability":    (0.4, 0.7), # jittery, moving around
    },
    "Overwhelmed": {
        "blink_rate":        (5,  15),   # reduced under high cognitive load
        "pupil_dilation":    (20, 50),   # high dilation = high mental effort
        "fixation_duration": (700, 2000),# prolonged staring, processing difficulty
        "saccade_rate":      (0.3, 1.5), # few saccades, "stuck" on problem
        "gaze_stability":    (0.6, 0.9), # centered but tense
    },
}


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices, w, h) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for blink detection.
    Formula from Soukupová & Čech (2016):
      EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Returns ~0.3 when eye is open, ~0.0 when closed.
    """
    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    # Convert normalized landmarks to pixel coords
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    
    # Vertical distances (p2-p6, p3-p5)
    v1 = dist(coords[1], coords[5])
    v2 = dist(coords[2], coords[4])
    # Horizontal distance (p1-p4)
    h_dist = dist(coords[0], coords[3])
    
    return (v1 + v2) / (2.0 * h_dist + 1e-6)


def iris_center(landmarks, iris_indices, w, h) -> Tuple[float, float]:
    """Returns (x, y) center of iris in pixel coordinates."""
    xs = [landmarks[i].x * w for i in iris_indices]
    ys = [landmarks[i].y * h for i in iris_indices]
    return (np.mean(xs), np.mean(ys))


def iris_diameter(landmarks, iris_indices, w, h) -> float:
    """
    Approximate iris diameter from the 5 iris landmarks.
    MediaPipe iris landmarks form a circle around the pupil.
    """
    coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices])
    # Diameter ≈ 2 * std of distances from center
    center = coords.mean(axis=0)
    dists = np.linalg.norm(coords - center, axis=1)
    return 2.0 * np.std(dists)


def gaze_angle(left_iris_center, right_iris_center, face_center) -> Tuple[float, float]:
    """
    Compute gaze deviation angles (horizontal, vertical) in degrees.
    Positive horizontal = looking right, positive vertical = looking down.
    """
    # Average of both irises
    gaze_x = (left_iris_center[0] + right_iris_center[0]) / 2.0
    gaze_y = (left_iris_center[1] + right_iris_center[1]) / 2.0
    
    # Deviation from face center
    dx = gaze_x - face_center[0]
    dy = gaze_y - face_center[1]
    
    # Convert to degrees (normalized by typical face width ~300px)
    horiz_angle = np.degrees(np.arctan2(dx, 300))
    vert_angle  = np.degrees(np.arctan2(dy, 300))
    
    return (horiz_angle, vert_angle)


# ─────────────────────────────────────────────────────────────
# EYETRACKER CLASS
# ─────────────────────────────────────────────────────────────

class EyeTracker:
    def __init__(self, window_size: int = 30):
        """
        Initialize eye tracker.
        
        Parameters
        ----------
        window_size : int
            Number of frames to use for rolling window metrics (default 30 = 1 sec at 30 FPS)
        """
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # enables iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Rolling buffers for metrics
        self.window_size = window_size
        self.blink_buffer       = deque(maxlen=window_size)  # bool per frame
        self.pupil_buffer       = deque(maxlen=window_size)  # diameter per frame
        self.fixation_buffer    = deque(maxlen=100)          # fixation durations (ms)
        self.saccade_buffer     = deque(maxlen=100)          # saccade events
        self.gaze_position_buffer = deque(maxlen=window_size)  # (x, y) per frame
        
        # State tracking
        self.baseline_pupil     = None   # set after first N frames
        self.last_iris_pos      = None   # for detecting saccades
        self.fixation_start     = None
        self.current_state      = "Focused"
        self.state_confidence   = 0.0
        self.frame_count        = 0
        
        # EAR threshold for blink detection
        self.EAR_THRESHOLD = 0.21  # empirically determined, may need tuning
        self.consecutive_closed = 0
        self.total_blinks = 0
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process one video frame and return eye metrics + attention state.
        
        Returns
        -------
        dict with keys:
          - attention_state: str (Focused/Drifting/Impulsive/Overwhelmed)
          - confidence: float (0-1)
          - metrics: dict of current metric values
          - landmarks: face mesh results (for visualization)
        """
        self.frame_count += 1
        h, w, _ = frame.shape
        
        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return {
                "attention_state": self.current_state,
                "confidence": 0.0,
                "metrics": {},
                "landmarks": None,
            }
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # ── Extract raw metrics ──────────────────────────────────
        
        # 1. Blink detection (EAR for both eyes)
        left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0
        
        is_blink = avg_ear < self.EAR_THRESHOLD
        self.blink_buffer.append(is_blink)
        
        if is_blink:
            self.consecutive_closed += 1
        else:
            if self.consecutive_closed >= 2:  # blink = at least 2 consecutive closed frames
                self.total_blinks += 1
            self.consecutive_closed = 0
        
        # 2. Pupil/iris tracking
        left_center  = iris_center(landmarks, LEFT_IRIS,  w, h)
        right_center = iris_center(landmarks, RIGHT_IRIS, w, h)
        left_diam    = iris_diameter(landmarks, LEFT_IRIS,  w, h)
        right_diam   = iris_diameter(landmarks, RIGHT_IRIS, w, h)
        avg_diam     = (left_diam + right_diam) / 2.0
        
        self.pupil_buffer.append(avg_diam)
        
        # Set baseline after first 30 frames
        if self.baseline_pupil is None and len(self.pupil_buffer) == self.window_size:
            self.baseline_pupil = np.mean(self.pupil_buffer)
        
        # 3. Gaze position and fixation/saccade detection
        face_center_x = w / 2.0
        face_center_y = h / 2.0
        
        gaze_x = (left_center[0] + right_center[0]) / 2.0
        gaze_y = (left_center[1] + right_center[1]) / 2.0
        self.gaze_position_buffer.append((gaze_x, gaze_y))
        
        # Detect saccades (rapid eye movements > 30 pixels/frame)
        if self.last_iris_pos is not None:
            movement = np.linalg.norm(np.array([gaze_x, gaze_y]) - np.array(self.last_iris_pos))
            if movement > 30:  # threshold for saccade
                self.saccade_buffer.append(time.time())
                # End current fixation
                if self.fixation_start is not None:
                    duration_ms = (time.time() - self.fixation_start) * 1000
                    self.fixation_buffer.append(duration_ms)
                    self.fixation_start = None
            else:
                # Continuing fixation or starting new one
                if self.fixation_start is None:
                    self.fixation_start = time.time()
        
        self.last_iris_pos = (gaze_x, gaze_y)
        
        # ── Compute rolling window metrics ───────────────────────
        
        metrics = {}
        
        # Blink rate (blinks per minute)
        if len(self.blink_buffer) == self.window_size:
            fps = 30  # assume 30 FPS, adjust if known
            duration_sec = self.window_size / fps
            metrics["blink_rate"] = (self.total_blinks / duration_sec) * 60.0
        else:
            metrics["blink_rate"] = 0.0
        
        # Pupil dilation (% change from baseline)
        if self.baseline_pupil is not None and len(self.pupil_buffer) > 0:
            current_avg = np.mean(list(self.pupil_buffer)[-10:])  # last 10 frames
            metrics["pupil_dilation"] = ((current_avg - self.baseline_pupil) / self.baseline_pupil) * 100.0
        else:
            metrics["pupil_dilation"] = 0.0
        
        # Fixation duration (median of recent fixations)
        if len(self.fixation_buffer) > 0:
            metrics["fixation_duration"] = float(np.median(self.fixation_buffer))
        else:
            metrics["fixation_duration"] = 0.0
        
        # Saccade rate (per second)
        recent_saccades = [t for t in self.saccade_buffer if time.time() - t < 2.0]  # last 2 seconds
        metrics["saccade_rate"] = len(recent_saccades) / 2.0
        
        # Gaze stability (how much gaze stays near center)
        if len(self.gaze_position_buffer) >= 10:
            recent_gaze = list(self.gaze_position_buffer)[-10:]
            center = np.array([face_center_x, face_center_y])
            deviations = [np.linalg.norm(np.array(g) - center) for g in recent_gaze]
            avg_deviation = np.mean(deviations)
            # Normalize: 0 deviation = 1.0 stability, 100px deviation = 0.5, 200px = 0.0
            metrics["gaze_stability"] = max(0.0, 1.0 - (avg_deviation / 200.0))
        else:
            metrics["gaze_stability"] = 1.0
        
        # ── Classify attention state ─────────────────────────────
        
        if self.frame_count < self.window_size:
            # Not enough data yet
            state = "Focused"
            confidence = 0.0
        else:
            state, confidence = self._classify_state(metrics)
            self.current_state = state
            self.state_confidence = confidence
        
        return {
            "attention_state": state,
            "confidence": confidence,
            "metrics": metrics,
            "landmarks": results.multi_face_landmarks[0],
            "blink_count": self.total_blinks,
            "avg_ear": avg_ear,
        }
    
    def _classify_state(self, metrics: Dict) -> Tuple[str, float]:
        """
        Multi-metric classifier: compute match score for each state,
        return the best match with confidence.
        """
        scores = {}
        
        for state_name, profile in STATE_PROFILES.items():
            score = 0.0
            count = 0
            
            for metric_name, (low, high) in profile.items():
                if metric_name not in metrics:
                    continue
                val = metrics[metric_name]
                # Score: 1.0 if in range, 0.0 if far outside, linear in between
                if low <= val <= high:
                    score += 1.0
                else:
                    # Distance from nearest bound
                    dist = min(abs(val - low), abs(val - high))
                    range_width = high - low
                    # Penalty proportional to how far out of range
                    penalty = max(0.0, 1.0 - (dist / (range_width * 2.0)))
                    score += penalty
                count += 1
            
            scores[state_name] = score / count if count > 0 else 0.0
        
        # Best match
        best_state = max(scores, key=scores.get)
        confidence = scores[best_state]
        
        return (best_state, confidence)
    
    def get_dataset_row(self, correct: int, response_time_s: float) -> Dict:
        """
        Format current eye tracking data as a row compatible with the
        ASSISTments-style dataset used by the ML model.
        
        This allows the eye tracker to feed directly into the LSTM pipeline.
        """
        if not self.pupil_buffer:
            return None
        
        metrics = {
            "blink_rate": (self.total_blinks / (self.frame_count / 30.0)) * 60.0 if self.frame_count > 0 else 0,
            "pupil_dilation": ((np.mean(self.pupil_buffer) - (self.baseline_pupil or 1.0)) / (self.baseline_pupil or 1.0)) * 100.0,
            "fixation_duration": float(np.median(self.fixation_buffer)) if self.fixation_buffer else 0.0,
            "saccade_rate": len([t for t in self.saccade_buffer if time.time() - t < 2.0]) / 2.0,
            "gaze_stability": 1.0,  # placeholder
        }
        
        return {
            "user_id": 1,  # placeholder
            "order_id": int(time.time()),
            "correct": correct,
            "response_time_s": response_time_s,
            "attempt_count": 1,
            "hint_count": 0,
            "hint_ratio": 0.0,
            "overlap_time": response_time_s,
            "attention_state_eye": self.current_state,
            "eye_blink_rate": metrics["blink_rate"],
            "eye_pupil_dilation": metrics["pupil_dilation"],
            "eye_fixation_duration": metrics["fixation_duration"],
            "eye_saccade_rate": metrics["saccade_rate"],
            "eye_gaze_stability": metrics["gaze_stability"],
        }


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def draw_overlay(frame, result, tracker):
    """Draw eye tracking visualization on frame."""
    h, w, _ = frame.shape
    
    # State badge
    state_colors = {
        "Focused":     (34, 197, 94),    # green
        "Drifting":    (245, 158, 11),   # orange
        "Impulsive":   (239, 68, 68),    # red
        "Overwhelmed": (139, 92, 246),   # purple
    }
    
    state = result["attention_state"]
    color = state_colors.get(state, (200, 200, 200))
    
    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, f"STATE: {state}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Conf: {result['confidence']:.0%}", (w - 150, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Metrics panel
    y = 80
    metrics = result.get("metrics", {})
    for name, val in metrics.items():
        label = name.replace("_", " ").title()
        if "rate" in name:
            text = f"{label}: {val:.1f} /min" if "blink" in name else f"{label}: {val:.1f} /s"
        elif "dilation" in name:
            text = f"{label}: {val:+.1f}%"
        elif "duration" in name:
            text = f"{label}: {val:.0f}ms"
        elif "stability" in name:
            text = f"{label}: {val:.2f}"
        else:
            text = f"{label}: {val:.1f}"
        
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        y += 25
    
    # Blink counter
    cv2.putText(frame, f"Blinks: {result.get('blink_count', 0)}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
    
    # Draw face mesh (optional, can be commented out for cleaner view)
    if result["landmarks"]:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=result["landmarks"],
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
        )
    
    return frame


# ─────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Eye tracking attention state detector")
    parser.add_argument("--video", type=str, help="Path to video file (default: webcam)")
    parser.add_argument("--integrate", action="store_true", help="Save output in dataset format")
    parser.add_argument("--output", type=str, default="eye_tracking_log.jsonl", help="Output file for --integrate")
    args = parser.parse_args()
    
    # Initialize
    tracker = EyeTracker(window_size=30)
    
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)  # webcam
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Eye Tracker initialized. Press 'q' to quit, 's' to save snapshot.")
    
    if args.integrate:
        log_file = open(args.output, "w")
        print(f"Logging data to {args.output}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = tracker.process_frame(frame)
        
        # Visualize
        frame_viz = draw_overlay(frame, result, tracker)
        
        cv2.imshow("Eye Tracker - Attention State Detection", frame_viz)
        
        # Save to log if integrating
        if args.integrate and result["metrics"]:
            row = tracker.get_dataset_row(correct=1, response_time_s=5.0)  # dummy values
            if row:
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame_viz)
            print(f"Saved {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if args.integrate:
        log_file.close()
        print(f"Logged {tracker.frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
