// src/utils/mastery.js

const KEY = "neurolearn_progress_v1";

function defaultState(moduleBank) {
  const state = {};
  moduleBank.forEach((m, idx) => {
    state[m.id] = {
      total: 10, // fixed set per module (change later per module)
      answered: 0,
      correct: 0,
      bestStreak: 0,
      avgRt: null, // average response time
      attentionCounts: {
        Focused: 0,
        Drifting: 0,
        Impulsive: 0,
        Overwhelmed: 0,
      },
      mastered: idx === 0 ? false : false, // mastery computed; locking handled separately
      unlocked: idx === 0, // only first module unlocked initially
      lastUpdated: Date.now(),
    };
  });
  return state;
}

export function loadProgress(moduleBank) {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaultState(moduleBank);
    const parsed = JSON.parse(raw);

    // ensure new modules get added if moduleBank changes
    const base = defaultState(moduleBank);
    for (const id of Object.keys(base)) {
      if (!parsed[id]) parsed[id] = base[id];
      else {
        // merge missing fields
        parsed[id] = { ...base[id], ...parsed[id] };
        parsed[id].attentionCounts = {
          ...base[id].attentionCounts,
          ...(parsed[id].attentionCounts || {}),
        };
      }
    }
    return parsed;
  } catch {
    return defaultState(moduleBank);
  }
}

export function saveProgress(progress) {
  localStorage.setItem(KEY, JSON.stringify(progress));
}

export function resetAllProgress(moduleBank) {
  const fresh = defaultState(moduleBank);
  saveProgress(fresh);
  return fresh;
}

/**
 * Mastery Criteria (demo-friendly):
 * - Minimum answered: 8 (out of total 10)
 * - Accuracy >= 75%
 * - Best streak >= 3
 * - Drifting + Overwhelmed <= 40% of answered (to avoid “mastery by random luck”)
 */
export function computeMastery(p) {
  const answered = p.answered;
  if (answered < 8) return { mastered: false, score: 0, reasons: ["Need more practice"] };

  const acc = answered === 0 ? 0 : (p.correct / answered) * 100;
  const streakOk = p.bestStreak >= 3;

  const drift = p.attentionCounts.Drifting || 0;
  const overwhelmed = p.attentionCounts.Overwhelmed || 0;
  const overloadRate = answered === 0 ? 1 : (drift + overwhelmed) / answered;

  const accOk = acc >= 75;
  const overloadOk = overloadRate <= 0.4;

  const reasons = [];
  if (!accOk) reasons.push("Accuracy below 75%");
  if (!streakOk) reasons.push("Need a streak of 3");
  if (!overloadOk) reasons.push("Too many drift/overwhelm moments");

  const mastered = accOk && streakOk && overloadOk;

  // score out of 100 (simple)
  const score = Math.round(
    0.6 * Math.min(100, acc) +
      0.25 * Math.min(100, (p.bestStreak / 5) * 100) +
      0.15 * Math.max(0, (1 - overloadRate) * 100)
  );

  return { mastered, score, reasons };
}

/**
 * Update progress after each answered question
 */
export function updateAfterAnswer(progress, moduleId, payload) {
  // payload: { correct, rt, attentionState, streak }
  const p = progress[moduleId];
  if (!p) return progress;

  const next = { ...progress };
  const updated = { ...p };

  if (updated.answered >= updated.total) {
    // module already complete; still update mastery stats if you want, but keep capped
    // For now: do nothing
    return progress;
  }

  updated.answered += 1;
  if (payload.correct) updated.correct += 1;
  updated.bestStreak = Math.max(updated.bestStreak, payload.streak || 0);

  // avg rt
  const prevAvg = updated.avgRt;
  updated.avgRt =
    prevAvg === null ? payload.rt : (prevAvg * (updated.answered - 1) + payload.rt) / updated.answered;

  // attention counts
  const ac = { ...(updated.attentionCounts || {}) };
  ac[payload.attentionState] = (ac[payload.attentionState] || 0) + 1;
  updated.attentionCounts = ac;

  // mastery compute
  const { mastered } = computeMastery(updated);
  updated.mastered = mastered;
  updated.lastUpdated = Date.now();

  next[moduleId] = updated;
  return next;
}

/**
 * Locking rule:
 * - Only next module unlocks when current module is mastered OR completed+mastered.
 * - First module unlocked by default.
 */
export function applyUnlockRules(progress, moduleBank) {
  const next = { ...progress };

  // ensure first always unlocked
  if (moduleBank[0]?.id) next[moduleBank[0].id].unlocked = true;

  for (let i = 0; i < moduleBank.length - 1; i++) {
    const curId = moduleBank[i].id;
    const nxtId = moduleBank[i + 1].id;

    const cur = next[curId];
    const nxt = next[nxtId];
    if (!nxt) continue;

    const shouldUnlockNext = Boolean(cur?.mastered);
    nxt.unlocked = shouldUnlockNext;
  }

  return next;
}