/**
 * useAttention — React hook that wires together:
 *   1. Browser eye tracking  (engines/eyeTracking.js)
 *   2. Behavioral classifier (engines/attentionModel.js)
 *   3. Fusion engine         (engines/attentionModel.js)
 *
 * Usage:
 *   const { eyeActive, eyeMetrics, attentionState, computeAttention, startEyeTracking } = useAttention();
 *
 *   // On answer submit:
 *   const result = computeAttention({ responseTime, correct, retries, recentErrors, idleTime });
 *   // result = { state: "Focused", confidence: 0.85, reasons: [...] }
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { EyeTracker } from "./eyeTracking.js";
import { detectAttentionFromBehavior, fuseAttentionStates } from "./attentionModel.js";

export function useAttention() {
  const trackerRef = useRef(null);
  const [eyeActive, setEyeActive] = useState(false);
  const [eyeState, setEyeState] = useState(null);       // { state, confidence, metrics }
  const [eyeMetrics, setEyeMetrics] = useState(null);
  const [attentionState, setAttentionState] = useState({ state: "Focused", confidence: 0.9, reasons: [] });

  /**
   * Initialize eye tracker & camera. Call once (e.g. on user click to grant permission).
   * Returns true if successfully started.
   */
  const startEyeTracking = useCallback(async () => {
    if (trackerRef.current) return true; // already running

    const tracker = new EyeTracker(30);
    const initOk = await tracker.init();
    if (!initOk) {
      console.warn("[useAttention] Eye tracker init failed");
      return false;
    }

    const camOk = await tracker.startCamera();
    if (!camOk) {
      console.warn("[useAttention] Camera access denied");
      return false;
    }

    tracker.onResult(({ state, confidence, metrics }) => {
      setEyeState({ state, confidence });
      setEyeMetrics(metrics);
    });

    trackerRef.current = tracker;
    setEyeActive(true);
    return true;
  }, []);

  /**
   * Compute fused attention state from behavioral signals + eye tracking.
   * Call after each answer submission.
   */
  const computeAttention = useCallback(
    ({ responseTime, correct, retries = 0, recentErrors = 0, idleTime = 0 }) => {
      // 1. Behavioral classification
      const behavioral = detectAttentionFromBehavior({
        responseTime,
        idleTime,
        correct,
        retries,
        recentErrors,
      });

      // 2. Fuse with eye tracking (if active)
      const fused = fuseAttentionStates(eyeState, behavioral);

      setAttentionState(fused);
      return fused;
    },
    [eyeState]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (trackerRef.current) {
        trackerRef.current.stop();
        trackerRef.current = null;
      }
    };
  }, []);

  return {
    eyeActive,
    eyeState,
    eyeMetrics,
    attentionState,
    computeAttention,
    startEyeTracking,
  };
}
