// src/pages/Game.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import { generateLangQuestion } from "../utils/api.js";
import { useAttention } from "../engines/useAttention.js";
import PixelSettingsModal from "../components/PixelSettingsModal";

import { moduleBank } from "../modules/moduleBank";
import {
  applyUnlockRules,
  computeMastery,
  loadProgress,
  saveProgress,
  updateAfterAnswer,
} from "../utils/mastery";

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function playTone(type) {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sine";
    osc.frequency.value = type === "correct" ? 640 : 220;
    gain.gain.value = 0.045;

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + 0.14);
    setTimeout(() => ctx.close?.(), 250);
  } catch {}
}

function modeBadge(state) {
  if (state === "Focused")
    return "border-[rgba(52,211,153,0.55)] bg-[rgba(52,211,153,0.12)]";
  if (state === "Drifting")
    return "border-[rgba(250,204,21,0.55)] bg-[rgba(250,204,21,0.10)]";
  if (state === "Impulsive")
    return "border-[rgba(232,121,249,0.55)] bg-[rgba(232,121,249,0.10)]";
  if (state === "Overwhelmed")
    return "border-[rgba(251,113,133,0.55)] bg-[rgba(251,113,133,0.10)]";
  return "border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)]";
}

function SparkBurst({ show }) {
  if (!show) return null;
  const sparks = Array.from({ length: 10 }, (_, i) => i);

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      {sparks.map((i) => (
        <span
          key={i}
          className="absolute left-1/2 top-1/2 h-2 w-2 rounded bg-[rgba(250,204,21,0.95)]"
          style={{
            transform: "translate(-50%, -50%)",
            animation: `spark-${i} 520ms ease-out forwards`,
          }}
        />
      ))}

      <style>{`
        ${sparks
          .map((i) => {
            const angle = (i / sparks.length) * Math.PI * 2;
            const dx = Math.round(Math.cos(angle) * 90);
            const dy = Math.round(Math.sin(angle) * 60);
            return `
              @keyframes spark-${i} {
                0%   { opacity: 0; transform: translate(-50%, -50%) scale(0.9); }
                10%  { opacity: 1; }
                100% { opacity: 0; transform: translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px)) scale(0.6); }
              }
            `;
          })
          .join("\n")}
      `}</style>
    </div>
  );
}

function TinyBar({ pct }) {
  return (
    <div className="mt-2 h-2 w-full rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
      <div
        className="h-2 bg-gradient-to-r from-[rgb(56,189,248)] to-[rgb(34,211,238)] transition-all"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export default function Game({ module, onBack, character: charProp }) {
  // Progress state (shared with dashboard via localStorage)
  const [progressState, setProgressState] = useState(() => loadProgress(moduleBank));

  // Character for story-wrapping (prop or default)
  const character = charProp?.name || charProp || "Dragon";

  // Eye tracking + attention fusion
  const {
    eyeActive, eyeMetrics, attentionState: fusedAttention,
    computeAttention, startEyeTracking,
  } = useAttention();

  // Gameplay
  const [difficulty, setDifficulty] = useState(2);
  const [loading, setLoading] = useState(false);

  const [mcq, setMcq] = useState(null);
  const [selected, setSelected] = useState(null);
  const [locked, setLocked] = useState(false);

  const [retries, setRetries] = useState(0);
  const [attention, setAttention] = useState({ state: "Focused", reasons: [] });
  const [showExplain, setShowExplain] = useState(false);
  const [hintIdx, setHintIdx] = useState(0);
  const [showHint, setShowHint] = useState(false);

  // Stats
  const [streak, setStreak] = useState(0);
  const [totalAnswered, setTotalAnswered] = useState(0);
  const [totalCorrect, setTotalCorrect] = useState(0);
  const [attentionHistory, setAttentionHistory] = useState([]);
  const [prevQuestions, setPrevQuestions] = useState([]);
  const [lastCorrect, setLastCorrect] = useState(null);
  const [lastRt, setLastRt] = useState(null);
  const recentErrorsRef = useRef(0);

  // Settings
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    sound: true,
    calm: false,
    autoCalm: true,
    sprintSeconds: 60,
  });

  // Sprint
  const [sprintMode, setSprintMode] = useState(false);
  const [timeLeft, setTimeLeft] = useState(settings.sprintSeconds);

  // Timers
  const startRef = useRef(Date.now());
  const idleTimerRef = useRef(null);
  const [idleTime, setIdleTime] = useState(0);

  const topic = useMemo(() => module?.geminiTopic ?? "general math", [module]);

  // Auto-start eye tracking on mount
  useEffect(() => { startEyeTracking(); }, [startEyeTracking]);
  const calmMode = settings.calm;
  const autoCalm = settings.autoCalm;

  // Sync body class for calm mode (cursor + reduced motion)
  useEffect(() => {
    if (calmMode) document.body.classList.add("calm");
    else document.body.classList.remove("calm");
    return () => document.body.classList.remove("calm");
  }, [calmMode]);

  const accuracy =
    totalAnswered === 0 ? 0 : Math.round((totalCorrect / totalAnswered) * 100);

  // Fake RPG stats
  const level = Math.max(1, Math.floor(totalCorrect / 5) + 1);
  const xp = totalCorrect * 20;
  const nextLevelXp = level * 120;
  const xpProgress = Math.min(100, Math.round((xp / nextLevelXp) * 100));

  // Typewriter
  const [typed, setTyped] = useState("");
  const typingRef = useRef({ timer: null });

  // Correct spark
  const [spark, setSpark] = useState(false);

  // Entrance animation (disabled in calm)
  const enterClass = calmMode ? "" : "animate-[enter_420ms_ease-out_forwards]";
  const enterStyle = `
    @keyframes enter {
      0% { opacity: 0; transform: translateY(10px); }
      100% { opacity: 1; transform: translateY(0px); }
    }
  `;

  // Module progress computed
  const moduleProgress = progressState?.[module.id];
  const totalQ = moduleProgress?.total ?? 10;
  const doneQ = moduleProgress?.answered ?? 0;
  const modulePct = Math.min(100, Math.round((doneQ / totalQ) * 100));
  const mastery = moduleProgress ? computeMastery(moduleProgress) : { mastered: false, score: 0, reasons: [] };
  const moduleComplete = doneQ >= totalQ;

  const resetTimers = () => {
    startRef.current = Date.now();
    setIdleTime(0);
    if (idleTimerRef.current) clearInterval(idleTimerRef.current);
    idleTimerRef.current = setInterval(() => setIdleTime((t) => t + 1), 1000);
  };

  const startTypewriter = (text) => {
    if (calmMode) {
      setTyped(text || "");
      return;
    }
    if (typingRef.current.timer) clearInterval(typingRef.current.timer);

    setTyped("");
    let i = 0;
    const speed = 14;

    typingRef.current.timer = setInterval(() => {
      i += 2;
      setTyped(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(typingRef.current.timer);
        typingRef.current.timer = null;
      }
    }, speed);
  };

  const load = async (nextDifficulty = difficulty) => {
    // STOP generating if module finished
    if (moduleComplete) return;

    setLoading(true);
    setLocked(false);
    setSelected(null);
    setShowExplain(false);
    setShowHint(false);
    setHintIdx(0);
    setSpark(false);

    const qNum = (moduleProgress?.answered ?? 0) + 1;
    const sessionAccuracy = totalAnswered === 0 ? 0 : totalCorrect / totalAnswered;

    try {
      let data;
      const useBackend = isBackendEnabled();

      if (useBackend) {
        try {
          data = await generateLangQuestion({
            character,
            topic,
            difficulty: nextDifficulty,
            attentionState: attention.state,
            questionNumber: qNum,
            totalQuestions: totalQ,
            sessionAccuracy,
            previousQuestions: prevQuestions.slice(-5),
            eyeMetrics: eyeActive && eyeMetrics ? {
              blink_rate: eyeMetrics.blink_rate,
              pupil_dilation: eyeMetrics.pupil_dilation,
              fixation_duration: eyeMetrics.fixation_duration,
              saccade_rate: eyeMetrics.saccade_rate,
              gaze_stability: eyeMetrics.gaze_stability,
            } : null,
          });
        } catch (err) {
          console.warn("Backend unavailable, using client Gemini:", err?.message);
          data = {};
        }
      } else {
        data = {};
      }

      setMcq(data);
      setRetries(0);
      resetTimers();
      startTypewriter(data?.question || "");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load(difficulty);
    return () => {
      idleTimerRef.current && clearInterval(idleTimerRef.current);
      if (typingRef.current.timer) clearInterval(typingRef.current.timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [module?.id]);

  useEffect(() => {
    if (!mcq) return;
    if (calmMode) {
      if (typingRef.current.timer) clearInterval(typingRef.current.timer);
      typingRef.current.timer = null;
      setTyped(mcq.question || "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [calmMode]);

  // Sprint timer
  useEffect(() => {
    if (!sprintMode) return;

    if (timeLeft <= 0) {
      setSprintMode(false);
      return;
    }

    const interval = setInterval(() => setTimeLeft((t) => t - 1), 1000);
    return () => clearInterval(interval);
  }, [sprintMode, timeLeft]);

  useEffect(() => {
    if (!sprintMode) setTimeLeft(settings.sprintSeconds);
  }, [settings.sprintSeconds, sprintMode]);

  const submit = () => {
    if (!mcq || selected === null || locked) return;
    if (sprintMode && timeLeft <= 0) return;
    if (moduleComplete) return;

    const rt = (Date.now() - startRef.current) / 1000;
    const correct = selected === mcq.correctIndex;

    if (settings.sound) playTone(correct ? "correct" : "wrong");

    // Track error burst for fusion
    if (!correct) recentErrorsRef.current++;
    else recentErrorsRef.current = Math.max(0, recentErrorsRef.current - 1);

    const result = computeAttention({
      responseTime: rt,
      correct,
      retries,
      recentErrors: recentErrorsRef.current,
      idleTime,
    });

    setAttention(result);
    setLocked(true);
    setShowExplain(true);

    if (correct && !calmMode) {
      setSpark(true);
      setTimeout(() => setSpark(false), 550);
    }

    // Auto calm
    if (autoCalm) {
      const shouldCalm =
        result.state === "Drifting" || result.state === "Overwhelmed";
      setSettings((s) => ({ ...s, calm: shouldCalm }));
    }

    // Stats (and for next-question context when using backend)
    setTotalAnswered((t) => t + 1);
    if (correct) setTotalCorrect((c) => c + 1);
    setLastCorrect(correct);
    setLastRt(rt);
    setAttentionHistory((prev) => [...prev.slice(-22), result.state]);
    setPrevQuestions((p) => [...p, mcq?.question || ""]);

    // Adaptive difficulty + streak + retries
    let nextStreak = 0;
    if (correct) {
      nextStreak = streak + 1;
      setStreak(nextStreak);

      if (result.state === "Focused") setDifficulty((d) => clamp(d + 1, 1, 5));
      if (result.state === "Drifting") setDifficulty((d) => clamp(d - 1, 1, 5));
      if (result.state === "Overwhelmed")
        setDifficulty((d) => clamp(d - 1, 1, 5));
    } else {
      setStreak(0);
      setRetries((r) => r + 1);
      if (result.state === "Overwhelmed" || result.state === "Drifting") {
        setDifficulty((d) => clamp(d - 1, 1, 5));
      }
    }

    // ======= MODULE PROGRESS UPDATE (score-based unlock) =======
    const updated = updateAfterAnswer(progressState, module.id, {
      correct,
      rt,
      attentionState: result.state,
      streak: correct ? nextStreak : 0,
    });

    const unlocked = applyUnlockRules(updated, moduleBank);
    setProgressState(unlocked);
    saveProgress(unlocked);
  };

  const next = () => {
    if (sprintMode && timeLeft <= 0) return;
    const p = progressState[module.id];
    if (p?.answered >= p?.total) return; // stop at limit
    load(difficulty);
  };

  const startSprint = () => {
    setSprintMode(true);
    setTimeLeft(settings.sprintSeconds);

    setTotalAnswered(0);
    setTotalCorrect(0);
    setAttentionHistory([]);
    setStreak(0);

    load(difficulty);
  };

  const optionStyle = (idx) => {
    if (!locked) {
      if (idx === selected) {
        return "border-[rgba(56,189,248,0.75)] bg-[rgba(56,189,248,0.12)] shadow-[0_0_0_2px_rgba(56,189,248,0.10)]";
      }
      return calmMode
        ? "border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.65)]"
        : "border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.65)] hover:bg-[rgba(13,26,58,0.85)]";
    }

    if (idx === mcq?.correctIndex)
      return "border-[rgba(52,211,153,0.75)] bg-[rgba(52,211,153,0.10)]";
    if (idx === selected && selected !== mcq?.correctIndex)
      return "border-[rgba(251,113,133,0.75)] bg-[rgba(251,113,133,0.10)]";
    return "border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.60)] opacity-80";
  };

  const wrapperClass = `min-h-screen ${calmMode ? "calm" : ""}`;

  const sprintProgress = sprintMode
    ? Math.max(0, Math.min(100, (timeLeft / settings.sprintSeconds) * 100))
    : 0;

  return (
    <div className={wrapperClass}>
      <style>{enterStyle}</style>

      {/* In-page controls (no navbar) */}
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-end gap-2 flex-wrap">
        <button
          onClick={onBack}
          className="px-4 py-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.85)] transition text-sm"
        >
          ← Back
        </button>
        <button
          onClick={startSprint}
          className="px-4 py-2 rounded-xl border border-[rgba(56,189,248,0.45)] bg-[rgba(56,189,248,0.10)] hover:bg-[rgba(56,189,248,0.16)] transition text-sm"
          title="Sprint"
        >
          Sprint
        </button>
        <button
          onClick={() => setSettingsOpen(true)}
          className="p-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.85)] transition"
          title="Settings"
          aria-label="Settings"
        >
          ⚙️
        </button>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className={`grid grid-cols-1 lg:grid-cols-3 gap-4 ${enterClass}`}>
          {/* Main */}
          <div className="lg:col-span-2">
            <div className="relative rounded-3xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.75)] shadow-xl overflow-hidden">
              <SparkBurst show={spark} />

              {/* Header */}
              <div className="p-6 md:p-8 bg-gradient-to-br from-[rgba(56,189,248,0.12)] to-transparent border-b border-[rgba(48,68,105,0.6)]">
                <div className="flex items-start justify-between gap-3 flex-wrap">
                  <div className="min-w-[240px]">
                    <div className="text-xs text-slate-300/80 tracking-[0.25em]">
                      QUEST • STORY MCQ
                    </div>

                    <div className="pixel-heading mt-3 text-2xl md:text-3xl">
                      {module.title}
                    </div>

                    <div className="mt-3 text-slate-200/90 leading-relaxed break-words">
                      Difficulty:{" "}
                      <span className="text-[rgb(56,189,248)] font-semibold">
                        {difficulty}
                      </span>{" "}
                      • Mode:{" "}
                      <span
                        className={`inline-flex items-center px-3 py-1 rounded-full border-2 ${modeBadge(
                          attention.state
                        )}`}
                      >
                        {attention.state}
                      </span>
                      {" "}• Buddy: <span className="font-semibold">{character}</span>
                    </div>

                    {/* Eye tracking + backend indicator */}
                    <div className="mt-2 flex items-center gap-2 text-xs text-slate-300/80 flex-wrap">
                      <span className={`inline-block h-2 w-2 rounded-full ${eyeActive ? "bg-[rgb(52,211,153)] animate-pulse" : "bg-[rgb(100,116,139)]"}`} />
                      <span>Eye: {eyeActive ? "On" : "Off"}</span>
                      {eyeActive && fusedAttention && (
                        <span>
                          👁 {fusedAttention.state} ({(fusedAttention.confidence * 100).toFixed(0)}%)
                        </span>
                      )}
                      {eyeActive && eyeMetrics && (
                        <span>
                          Blink {eyeMetrics.blink_rate?.toFixed(0)}/min • Pupil {eyeMetrics.pupil_dilation?.toFixed(1)}%
                        </span>
                      )}
                      {isBackendEnabled() && (
                        <span className="text-[rgb(56,189,248)]">• Backend (state → next Q)</span>
                      )}
                    </div>

                    {/* Module progress */}
                    <div className="mt-4 rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] p-4">
                      <div className="flex items-center justify-between text-sm text-slate-200/90">
                        <span>
                          Module Progress:{" "}
                          <span className="font-semibold">
                            {doneQ}/{totalQ}
                          </span>
                        </span>
                        <span>
                          Mastery:{" "}
                          <span className="font-semibold text-[rgb(56,189,248)]">
                            {mastery.score}%
                          </span>
                        </span>
                      </div>
                      <TinyBar pct={modulePct} />
                      <div className="mt-2 text-xs text-slate-300/80 leading-relaxed break-words">
                        {mastery.mastered
                          ? "✅ Mastered! Go back to dashboard to unlock next quest."
                          : mastery.reasons?.length
                          ? `To master: ${mastery.reasons.join(" • ")}`
                          : "Keep going…"}
                      </div>
                    </div>
                  </div>

                  {/* Sprint */}
                  <div className="w-full md:w-[280px] rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] p-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-200/90">⚡ Sprint</span>
                      <span className="text-slate-200/90">
                        {sprintMode ? `${timeLeft}s` : `${settings.sprintSeconds}s`}
                      </span>
                    </div>
                    <div className="mt-3 h-2 w-full rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
                      <div
                        className="h-2 bg-gradient-to-r from-[rgb(56,189,248)] to-[rgb(34,211,238)] transition-all"
                        style={{ width: `${sprintMode ? sprintProgress : 0}%` }}
                      />
                    </div>
                    <div className="mt-2 text-xs text-slate-300/80">
                      {sprintMode ? "Sprint running…" : "Start sprint from navbar"}
                    </div>
                  </div>
                </div>
              </div>

              {/* Body */}
              <div className="p-6 md:p-8">
                {moduleComplete ? (
                  <div className="rounded-3xl border-2 border-[rgba(52,211,153,0.55)] bg-[rgba(52,211,153,0.10)] p-5">
                    <div className="pixel-heading text-base md:text-lg">
                      MODULE COMPLETE ✅
                    </div>
                    <div className="mt-2 text-slate-100/90 leading-relaxed break-words">
                      You reached the question limit for this module. Go back to dashboard
                      to unlock the next quest (only if mastery is achieved).
                    </div>
                    <div className="mt-4 flex gap-2 flex-wrap">
                      <button onClick={onBack} className="btn-pixel px-6 py-3 rounded-xl">
                        Back to Dashboard
                      </button>
                    </div>
                  </div>
                ) : loading || !mcq ? (
                  <div className="text-slate-200/90">
                    <div className="text-sm">Generating a quest challenge…</div>
                    <div className="mt-3 h-2 w-full rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
                      <div className="h-2 w-1/3 bg-[rgb(56,189,248)] animate-pulse" />
                    </div>
                  </div>
                ) : (
                  <>
                    {mcq.reasoning && (
                      <div className="mb-4 rounded-2xl border-2 border-[rgba(56,189,248,0.35)] bg-[rgba(56,189,248,0.06)] p-3">
                        <div className="text-xs text-[rgb(56,189,248)] font-semibold tracking-[0.2em]">WHY THIS QUESTION</div>
                        <div className="mt-1 text-sm text-slate-200/90 leading-relaxed break-words">
                          {mcq.reasoning}
                        </div>
                      </div>
                    )}

                    <div className="text-xs text-slate-300/80 tracking-[0.25em]">
                      OBJECTIVE
                    </div>

                    <div className="mt-3 text-xl md:text-2xl font-bold text-slate-100 leading-relaxed break-words">
                      {typed}
                      {!calmMode && typed?.length < (mcq.question?.length || 0) && (
                        <span className="inline-block w-[10px] ml-1 animate-pulse">
                          ▍
                        </span>
                      )}
                    </div>

                    <div className="mt-6 grid grid-cols-1 gap-3">
                      {(mcq.options ?? []).map((opt, idx) => (
                        <button
                          key={idx}
                          onClick={() => !locked && setSelected(idx)}
                          className={`text-left rounded-3xl border-2 p-4 md:p-5 transition active:scale-[0.99] focus:outline-none focus:ring-2 focus:ring-[rgba(56,189,248,0.35)] ${optionStyle(
                            idx
                          )}`}
                        >
                          <div className="flex items-start gap-4">
                            <div className="h-9 w-9 rounded-xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] flex items-center justify-center text-sm font-bold">
                              {String.fromCharCode(65 + idx)}
                            </div>
                            <div className="text-base md:text-lg text-slate-100 leading-relaxed break-words">
                              {opt}
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>

                    {/* Hints */}
                    {showHint && mcq?.hints && (
                      <div className="mt-4 rounded-2xl border-2 border-[rgba(250,204,21,0.35)] bg-[rgba(250,204,21,0.05)] p-4">
                        <div className="text-sm text-[rgb(250,204,21)] font-semibold">Hint {hintIdx}</div>
                        <div className="mt-1 text-slate-200/90">
                          {hintIdx === 1 ? (mcq.hints.hint_1 || mcq.hints[0]) : (mcq.hints.hint_2 || mcq.hints[1])}
                        </div>
                      </div>
                    )}

                    <div className="mt-6 flex gap-3 flex-wrap items-center">
                      {!locked ? (
                        <>
                          <button
                            className="btn-pixel px-6 py-3 rounded-xl disabled:opacity-50"
                            onClick={submit}
                            disabled={selected === null}
                          >
                            Submit
                          </button>
                          {mcq?.hints && hintIdx < 2 && (
                            <button
                              onClick={() => { setHintIdx((h) => h + 1); setShowHint(true); }}
                              className="px-5 py-3 rounded-xl border-2 border-[rgba(250,204,21,0.4)] bg-[rgba(250,204,21,0.06)] hover:bg-[rgba(250,204,21,0.12)] transition text-sm"
                            >
                              💡 Hint
                            </button>
                          )}
                        </>
                      ) : (
                        <button
                          className="btn-pixel px-6 py-3 rounded-xl disabled:opacity-50"
                          onClick={next}
                        >
                          Next
                        </button>
                      )}

                      <button
                        className="px-5 py-3 rounded-xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.85)] transition"
                        onClick={() => setShowExplain((v) => !v)}
                      >
                        {showExplain ? "Hide Solution" : "Show Solution"}
                      </button>

                      <div className="text-sm text-slate-300/80">
                        Tip: pick → submit. Slow + steady beats spam-clicking.
                      </div>
                    </div>

                    {showExplain && (
                      <div
                        className={`mt-6 rounded-3xl border-2 border-[rgba(56,189,248,0.35)] bg-[rgba(56,189,248,0.08)] p-5 ${
                          calmMode ? "" : "animate-[enter_360ms_ease-out_forwards]"
                        }`}
                      >
                        <div className="flex items-center justify-between flex-wrap gap-2">
                          <div className="pixel-heading text-base md:text-lg">
                            SOLUTION
                          </div>

                          <div className="text-sm text-slate-200/90">
                            Correct:{" "}
                            <span className="text-[rgb(56,189,248)] font-bold">
                              {String.fromCharCode(65 + mcq.correctIndex)}
                            </span>
                          </div>
                        </div>

                        <div className="mt-3 text-slate-100/90 leading-relaxed break-words">
                          {mcq.explanation || "No explanation provided."}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-4">
            <div className="rounded-3xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.75)] shadow-xl overflow-hidden">
              <div className="p-5 bg-[rgba(13,26,58,0.55)] border-b border-[rgba(48,68,105,0.6)]">
                <div className="flex items-center gap-3">
                  <div className="h-12 w-12 rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] flex items-center justify-center text-xl">
                    🧙
                  </div>
                  <div>
                    <div className="font-bold">Parth</div>
                    <div className="text-sm text-slate-300/80">
                      Level {level} • {module.id}
                    </div>
                  </div>
                </div>

                <div className="mt-4">
                  <div className="flex items-center justify-between text-xs text-slate-300/80">
                    <span>XP</span>
                    <span>
                      {xp} / {nextLevelXp}
                    </span>
                  </div>
                  <div className="mt-2 h-2 rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
                    <div
                      className="h-2 bg-gradient-to-r from-[rgb(56,189,248)] to-[rgb(34,211,238)] transition-all"
                      style={{ width: `${xpProgress}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="p-5 grid grid-cols-2 gap-3">
                <div className="rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] p-3">
                  <div className="text-xs text-slate-300/80">Accuracy</div>
                  <div className="mt-1 text-lg font-bold">{accuracy}%</div>
                </div>
                <div className="rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] p-3">
                  <div className="text-xs text-slate-300/80">Streak</div>
                  <div className="mt-1 text-lg font-bold">{streak} 🔥</div>
                </div>
                <div className="rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] p-3">
                  <div className="text-xs text-slate-300/80">Idle</div>
                  <div className="mt-1 text-lg font-bold">{idleTime}s</div>
                </div>
                <div className="rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] p-3">
                  <div className="text-xs text-slate-300/80">Retries</div>
                  <div className="mt-1 text-lg font-bold">{retries}</div>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.75)] shadow-xl p-5">
              <div className="pixel-heading text-base md:text-lg">Why this mode?</div>
              {attention.reasons?.length ? (
                <ul className="mt-3 text-sm text-slate-200/90 list-disc pl-5 space-y-1 break-words">
                  {attention.reasons.slice(0, 5).map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              ) : (
                <div className="mt-3 text-sm text-slate-300/80">
                  Submit once to see signals.
                </div>
              )}

              <div className="mt-4 rounded-2xl border-2 border-[rgba(56,189,248,0.35)] bg-[rgba(56,189,248,0.08)] p-3 text-sm text-slate-100/90 leading-relaxed break-words">
                Focused → harder • Drifting/Overwhelmed → easier • Impulsive → slow down
              </div>
            </div>

            <div className="rounded-3xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.75)] shadow-xl p-5">
              <div className="pixel-heading text-base md:text-lg">Attention Trend</div>
              <div className="mt-3 flex gap-1 flex-wrap">
                {attentionHistory.length ? (
                  attentionHistory.map((a, i) => (
                    <div
                      key={i}
                      title={a}
                      className={`h-3 w-3 rounded-full ${
                        a === "Focused"
                          ? "bg-[rgb(52,211,153)]"
                          : a === "Drifting"
                          ? "bg-[rgb(250,204,21)]"
                          : a === "Impulsive"
                          ? "bg-[rgb(232,121,249)]"
                          : "bg-[rgb(251,113,133)]"
                      }`}
                    />
                  ))
                ) : (
                  <div className="text-sm text-slate-300/80">
                    Answer a few to see the trend.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Stars */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute inset-0 opacity-[0.22] animate-pulse">
          <div className="absolute left-[12%] top-[20%] h-1 w-1 bg-white rounded" />
          <div className="absolute left-[40%] top-[12%] h-1 w-1 bg-white rounded" />
          <div className="absolute left-[72%] top-[24%] h-1 w-1 bg-white rounded" />
          <div className="absolute left-[85%] top-[10%] h-1 w-1 bg-white rounded" />
          <div className="absolute left-[20%] top-[72%] h-1 w-1 bg-white rounded" />
          <div className="absolute left-[58%] top-[78%] h-1 w-1 bg-white rounded" />
        </div>
      </div>

      <PixelSettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        setSettings={setSettings}
      />
    </div>
  );
}