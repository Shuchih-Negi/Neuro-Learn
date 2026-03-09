/**
 * Browser-based eye tracking using MediaPipe FaceLandmarker.
 * Extracts: blink rate, pupil dilation, fixation duration, saccade rate, gaze stability.
 * Classifies into: Focused | Drifting | Impulsive | Overwhelmed
 */

const LEFT_IRIS = [474, 475, 476, 477];
const RIGHT_IRIS = [469, 470, 471, 472];
const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];

const STATE_PROFILES = {
  Focused:     { blink_rate: [12,25],  pupil_dilation: [0,20],   fixation_duration: [250,700],  saccade_rate: [1.0,3.0],  gaze_stability: [0.7,1.0] },
  Drifting:    { blink_rate: [25,45],  pupil_dilation: [-15,5],  fixation_duration: [50,250],   saccade_rate: [3.0,6.0],  gaze_stability: [0.3,0.6] },
  Impulsive:   { blink_rate: [30,55],  pupil_dilation: [10,30],  fixation_duration: [30,150],   saccade_rate: [5.0,10.0], gaze_stability: [0.4,0.7] },
  Overwhelmed: { blink_rate: [5,15],   pupil_dilation: [20,50],  fixation_duration: [700,2000], saccade_rate: [0.3,1.5],  gaze_stability: [0.6,0.9] },
};

function dist(a, b) {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);
}

function ear(lm, indices, w, h) {
  const c = indices.map(i => [lm[i].x * w, lm[i].y * h]);
  return (dist(c[1], c[5]) + dist(c[2], c[4])) / (2 * dist(c[0], c[3]) + 1e-6);
}

function irisDiam(lm, indices, w, h) {
  const coords = indices.map(i => [lm[i].x * w, lm[i].y * h]);
  const cx = coords.reduce((s, c) => s + c[0], 0) / coords.length;
  const cy = coords.reduce((s, c) => s + c[1], 0) / coords.length;
  const dists = coords.map(c => dist(c, [cx, cy]));
  return 2 * (dists.reduce((s, d) => s + d, 0) / dists.length);
}

function irisCenter(lm, indices, w, h) {
  const x = indices.reduce((s, i) => s + lm[i].x * w, 0) / indices.length;
  const y = indices.reduce((s, i) => s + lm[i].y * h, 0) / indices.length;
  return [x, y];
}

export class EyeTracker {
  constructor(windowSize = 30) {
    this.W = windowSize;
    this.pupilBuf = [];
    this.gazeBuf = [];
    this.fixBuf = [];
    this.saccBuf = [];
    this.blinkWin = [];
    this.baseline = null;
    this.lastPos = null;
    this.fixStart = null;
    this.state = "Focused";
    this.conf = 0.5;
    this.frame = 0;
    this.blinks = 0;
    this.consec = 0;
    this.EAR_THRESHOLD = 0.21;
    this.landmarker = null;
    this.video = null;
    this.running = false;
    this._onResult = null;
  }

  async init() {
    try {
      const vision = await import("@mediapipe/tasks-vision");
      const { FaceLandmarker, FilesetResolver } = vision;

      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      this.landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU",
        },
        outputFaceBlendshapes: false,
        runningMode: "VIDEO",
        numFaces: 1,
      });

      return true;
    } catch (e) {
      console.warn("Eye tracker init failed:", e);
      return false;
    }
  }

  async startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      this.video = document.createElement("video");
      this.video.srcObject = stream;
      this.video.setAttribute("playsinline", "");
      await this.video.play();
      this.running = true;
      this._loop();
      return true;
    } catch (e) {
      console.warn("Camera access failed:", e);
      return false;
    }
  }

  _loop() {
    if (!this.running || !this.landmarker || !this.video) return;

    const now = performance.now();
    const result = this.landmarker.detectForVideo(this.video, now);

    if (result?.faceLandmarks?.length) {
      const metrics = this._processLandmarks(result.faceLandmarks[0], this.video.videoWidth, this.video.videoHeight);
      if (this._onResult) this._onResult({ state: this.state, confidence: this.conf, metrics });
    }

    requestAnimationFrame(() => this._loop());
  }

  _processLandmarks(landmarks, w, h) {
    this.frame++;
    const lm = landmarks;

    // Blink detection
    const earVal = (ear(lm, LEFT_EYE, w, h) + ear(lm, RIGHT_EYE, w, h)) / 2;
    if (earVal < this.EAR_THRESHOLD) {
      this.consec++;
      this.blinkWin.push(0);
    } else {
      if (this.consec >= 2) { this.blinks++; this.blinkWin.push(1); }
      else this.blinkWin.push(0);
      this.consec = 0;
    }
    if (this.blinkWin.length > this.W) this.blinkWin.shift();

    // Pupil diameter
    const diam = (irisDiam(lm, LEFT_IRIS, w, h) + irisDiam(lm, RIGHT_IRIS, w, h)) / 2;
    this.pupilBuf.push(diam);
    if (this.pupilBuf.length > this.W) this.pupilBuf.shift();
    if (!this.baseline && this.pupilBuf.length === this.W) {
      this.baseline = this.pupilBuf.reduce((s, v) => s + v, 0) / this.W;
    }

    // Gaze position
    const lc = irisCenter(lm, LEFT_IRIS, w, h);
    const rc = irisCenter(lm, RIGHT_IRIS, w, h);
    const gx = (lc[0] + rc[0]) / 2;
    const gy = (lc[1] + rc[1]) / 2;
    this.gazeBuf.push([gx, gy]);
    if (this.gazeBuf.length > this.W) this.gazeBuf.shift();

    // Saccade detection
    const now = Date.now();
    if (this.lastPos) {
      const mv = dist([gx, gy], this.lastPos);
      if (mv > 30) {
        this.saccBuf.push(now);
        if (this.fixStart) { this.fixBuf.push(now - this.fixStart); this.fixStart = null; }
      } else if (!this.fixStart) {
        this.fixStart = now;
      }
    }
    this.lastPos = [gx, gy];
    if (this.fixBuf.length > 100) this.fixBuf.shift();
    this.saccBuf = this.saccBuf.filter(t => now - t < 2000);

    // Compute metrics
    const fps = 30;
    const wSec = this.blinkWin.length / fps;
    const metrics = {
      blink_rate: wSec > 0 ? (this.blinkWin.reduce((s, v) => s + v, 0) / wSec) * 60 : 0,
      pupil_dilation: this.baseline ? ((this.pupilBuf.slice(-10).reduce((s, v) => s + v, 0) / Math.min(10, this.pupilBuf.length) - this.baseline) / this.baseline) * 100 : 0,
      fixation_duration: this.fixBuf.length > 0 ? this.fixBuf.slice().sort((a, b) => a - b)[Math.floor(this.fixBuf.length / 2)] : 300,
      saccade_rate: this.saccBuf.length / 2,
      gaze_stability: this.gazeBuf.length >= 10
        ? Math.max(0, 1 - this.gazeBuf.slice(-10).reduce((s, g) => s + dist(g, [w / 2, h / 2]), 0) / 10 / 200)
        : 1.0,
    };

    if (this.frame >= this.W) {
      [this.state, this.conf] = this._classify(metrics);
    }

    return metrics;
  }

  _classify(m) {
    const scores = {};
    for (const [state, profile] of Object.entries(STATE_PROFILES)) {
      let tot = 0, n = 0;
      for (const [key, [lo, hi]] of Object.entries(profile)) {
        if (!(key in m)) continue;
        const v = m[key];
        tot += (lo <= v && v <= hi) ? 1 : Math.max(0, 1 - Math.min(Math.abs(v - lo), Math.abs(v - hi)) / ((hi - lo || 1) * 2));
        n++;
      }
      scores[state] = n > 0 ? tot / n : 0;
    }
    const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
    return [best[0], best[1]];
  }

  onResult(callback) { this._onResult = callback; }

  getState() { return { state: this.state, confidence: this.conf }; }

  stop() {
    this.running = false;
    if (this.video?.srcObject) {
      this.video.srcObject.getTracks().forEach(t => t.stop());
    }
    if (this.landmarker) this.landmarker.close();
  }
}
