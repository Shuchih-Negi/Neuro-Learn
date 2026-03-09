/**
 * Rule-based behavioral attention classifier.
 * Mirrors the LSTM pipeline logic for frontend use.
 * Fuses eye tracking + behavioral signals.
 */

export function detectAttentionFromBehavior({ responseTime, idleTime, correct, retries, recentErrors }) {
  const reasons = [];

  if (responseTime < 2.2) reasons.push("Super fast answer");
  if (responseTime > 12) reasons.push("Slow answer");
  if (idleTime > 7) reasons.push("Idle spike");
  if (retries >= 2) reasons.push("Many retries");
  if (!correct) reasons.push("Wrong answer");
  if (recentErrors >= 2) reasons.push("Error burst");

  // Classification rules (from LSTM rule-based fallback)
  if (responseTime < 2.2 && !correct) return { state: "Impulsive", reasons, confidence: 0.85 };
  if (recentErrors >= 2 && responseTime < 4) return { state: "Impulsive", reasons, confidence: 0.8 };
  if (idleTime > 7 || responseTime > 12) return { state: "Drifting", reasons, confidence: 0.8 };
  if (retries >= 2 && !correct) return { state: "Overwhelmed", reasons, confidence: 0.85 };
  if (!correct && responseTime > 20) return { state: "Overwhelmed", reasons, confidence: 0.8 };
  return { state: "Focused", reasons, confidence: 0.9 };
}

/**
 * Fusion engine: combines eye tracking + behavioral signals
 * Eye weight = 0.4, Behavior weight = 0.6
 * Only uses eye data when confidence >= 0.55
 */
export function fuseAttentionStates(eyeState, behaviorState) {
  const EYE_WEIGHT = 0.4;
  const EYE_CONF_THRESHOLD = 0.55;

  if (!eyeState || eyeState.confidence < EYE_CONF_THRESHOLD) {
    return behaviorState;
  }

  const states = ["Focused", "Drifting", "Impulsive", "Overwhelmed"];

  // Simple voting with confidence weighting
  const eyeScore = {};
  const behScore = {};
  states.forEach(s => {
    eyeScore[s] = s === eyeState.state ? eyeState.confidence : 0;
    behScore[s] = s === behaviorState.state ? behaviorState.confidence : 0;
  });

  const fused = {};
  states.forEach(s => {
    fused[s] = EYE_WEIGHT * eyeScore[s] + (1 - EYE_WEIGHT) * behScore[s];
  });

  const best = Object.entries(fused).sort((a, b) => b[1] - a[1])[0];
  return {
    state: best[0],
    confidence: best[1],
    reasons: [...(behaviorState.reasons || [])],
  };
}
