import { useEffect, useRef, useState } from "react";
import { generateQuestion } from "../utils/api";
import { useAttention } from "../engines/useAttention.js";
import { updateProgress } from "../utils/api";

function clamp(n, a, b) { return Math.max(a, Math.min(b, n)); }

export default function TestPage({ section, character, onFinish, onBack }) {
  const TOTAL = 10;
  const [qNum, setQNum] = useState(0); // 0 = loading first
  const [mcq, setMcq] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [locked, setLocked] = useState(false);
  const [showExplain, setShowExplain] = useState(false);
  const [hintIdx, setHintIdx] = useState(0);
  const [showHint, setShowHint] = useState(false);

  // Typewriter
  const [typed, setTyped] = useState("");
  const typingRef = useRef(null);

  // Eye tracking + attention fusion
  const {
    eyeActive, eyeMetrics, attentionState: fusedAttention,
    computeAttention, startEyeTracking,
  } = useAttention();

  // Stats
  const [difficulty, setDifficulty] = useState(2);
  const [totalCorrect, setTotalCorrect] = useState(0);
  const [streak, setStreak] = useState(0);
  const [retries, setRetries] = useState(0);
  const [attentionState, setAttentionState] = useState("Focused");
  const [attentionHistory, setAttentionHistory] = useState([]);
  const [prevQuestions, setPrevQuestions] = useState([]);
  const recentErrorsRef = useRef(0);

  // Timers
  const startRef = useRef(Date.now());
  const idleRef = useRef(null);
  const [idleTime, setIdleTime] = useState(0);

  const accuracy = qNum > 0 ? totalCorrect / qNum : 0;

  const resetTimers = () => {
    startRef.current = Date.now();
    setIdleTime(0);
    if (idleRef.current) clearInterval(idleRef.current);
    idleRef.current = setInterval(() => setIdleTime((t) => t + 1), 1000);
  };

  const startTypewriter = (text) => {
    if (typingRef.current) clearInterval(typingRef.current);
    setTyped("");
    let i = 0;
    typingRef.current = setInterval(() => {
      i += 2;
      setTyped(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(typingRef.current);
        typingRef.current = null;
      }
    }, 14);
  };

  const loadQuestion = async (diff = difficulty) => {
    setLoading(true);
    setSelected(null);
    setLocked(false);
    setShowExplain(false);
    setShowHint(false);
    setHintIdx(0);

    try {
      const data = await generateQuestion({
        character: character?.name || character || "Dragon",
        topic: section.topic,
        difficulty: diff,
        attentionState,
        questionNumber: qNum + 1,
        totalQuestions: TOTAL,
        sessionAccuracy: Math.round(accuracy * 100),
        previousQuestions: prevQuestions.slice(-4),
        eyeMetrics,
      });
      setMcq(data);
      resetTimers();
      startTypewriter(data?.question || "");
    } catch {
      // Fallback
      setMcq({
        question: `Solve for x: 3x + 5 = 20`,
        options: ["A) x = 5", "B) x = 4", "C) x = 6", "D) x = 3"],
        correctIndex: 0,
        explanation: "3x = 15, x = 5",
        hints: { hint_1: "Isolate x", hint_2: "Subtract 5 from both sides" },
      });
      resetTimers();
      startTypewriter("Solve for x: 3x + 5 = 20");
    } finally {
      setLoading(false);
    }
  };

  // Auto-start eye tracking
  useEffect(() => { startEyeTracking(); }, [startEyeTracking]);

  useEffect(() => {
    loadQuestion();
    return () => {
      if (idleRef.current) clearInterval(idleRef.current);
      if (typingRef.current) clearInterval(typingRef.current);
    };
  }, []);

  const submit = () => {
    if (!mcq || selected === null || locked) return;
    const rt = (Date.now() - startRef.current) / 1000;
    const ci = mcq.correctIndex ?? mcq.correct_index ?? 0;
    const correct = selected === ci;

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
    const state = result.state;

    setAttentionState(state);
    setAttentionHistory((h) => [...h, state]);
    setLocked(true);
    setShowExplain(true);

    const newQ = qNum + 1;
    setQNum(newQ);
    if (correct) {
      setTotalCorrect((c) => c + 1);
      setStreak((s) => s + 1);
    } else {
      setStreak(0);
      setRetries((r) => r + 1);
    }

    // Adaptive difficulty
    if (state === "Focused" && correct) setDifficulty((d) => clamp(d + 1, 1, 5));
    else if (state === "Overwhelmed") setDifficulty((d) => clamp(d - 1, 1, 5));
    else if (state === "Drifting") setDifficulty((d) => clamp(d - 1, 1, 5));

    setPrevQuestions((p) => [...p, mcq.question]);

    // Update backend progress
    updateProgress({
      sectionId: section.id,
      correct,
      responseTime: rt,
      attentionState: state,
      xpEarned: correct ? 20 : 5,
    }).catch(() => {});
  };

  const next = () => {
    if (qNum >= TOTAL) {
      onFinish({ totalCorrect, totalQuestions: TOTAL, attentionHistory });
      return;
    }
    loadQuestion(difficulty);
  };

  const optionStyle = (idx) => {
    if (!locked) {
      return idx === selected
        ? "border-[rgba(56,189,248,0.75)] bg-[rgba(56,189,248,0.12)]"
        : "border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.65)] hover:bg-[rgba(13,26,58,0.85)]";
    }
    if (idx === mcq?.correct_index) return "border-[rgba(52,211,153,0.75)] bg-[rgba(52,211,153,0.10)]";
    if (idx === selected && selected !== mcq?.correct_index) return "border-[rgba(251,113,133,0.75)] bg-[rgba(251,113,133,0.10)]";
    return "border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.60)] opacity-80";
  };

  return (
    <div className="min-h-screen text-slate-100">
      {/* Top bar */}
        <div className="sticky top-0 z-50 backdrop-blur-md border-b border-white/[0.06] bg-[rgba(15,23,42,0.82)]">
        <div className="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
          <button onClick={onBack} className="px-4 py-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.85)] transition text-sm">
            ← Roadmap
          </button>
          <div className="flex items-center gap-4">
            {/* Eye tracking indicator */}
            <span className={`inline-block h-2 w-2 rounded-full ${eyeActive ? "bg-[rgb(52,211,153)] animate-pulse" : "bg-[rgb(100,116,139)]"}`} />
            <span className="text-xs text-slate-300/80">
              {eyeActive ? `👁 ${fusedAttention.state}` : "Eye: Off"}
            </span>
            <img src={character.image} alt={character.name} className="w-8 h-8 object-contain rounded-lg" />
            <span className="text-sm">{Math.min(qNum, TOTAL)}/{TOTAL}</span>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Progress */}
        <div className="h-2 w-full rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
          <div className="h-2 bg-gradient-to-r from-[rgb(56,189,248)] to-[rgb(94,234,212)] transition-all" style={{ width: `${(Math.min(qNum, TOTAL) / TOTAL) * 100}%` }} />
        </div>

        <div className="mt-8">
          <div className="rounded-3xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.75)] shadow-xl overflow-hidden">
            {/* Header */}
            <div className="p-6 bg-gradient-to-br from-[rgba(56,189,248,0.12)] to-transparent border-b border-[rgba(48,68,105,0.6)]">
              <div className="pixel-heading text-xl">{section.title}</div>
              <div className="mt-2 text-sm text-slate-300/80">
                Question {Math.min(qNum + (locked ? 0 : 1), TOTAL)} of {TOTAL} • Difficulty {difficulty}/5
              </div>
            </div>

            {/* Body */}
            <div className="p-6 md:p-8">
              {loading ? (
                <div className="py-10">
                  <div className="text-sm text-slate-200/90">Generating quest challenge...</div>
                  <div className="mt-3 h-2 w-full rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
                    <div className="h-2 w-1/3 bg-[rgb(56,189,248)] animate-pulse" />
                  </div>
                </div>
              ) : (
                <>
                  {/* Question */}
                  <div className="text-xl md:text-2xl font-bold text-slate-100 leading-relaxed">
                    {typed}
                    {typed?.length < (mcq?.question?.length || 0) && (
                      <span className="inline-block w-[10px] ml-1 animate-pulse">▍</span>
                    )}
                  </div>

                  {/* Options */}
                  <div className="mt-6 grid grid-cols-1 gap-3">
                    {(mcq?.options ?? []).map((opt, idx) => (
                      <button
                        key={idx}
                        onClick={() => !locked && setSelected(idx)}
                        className={`text-left rounded-2xl border-2 p-4 transition ${optionStyle(idx)}`}
                      >
                        <div className="flex items-start gap-4">
                          <div className="h-9 w-9 rounded-xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] flex items-center justify-center text-sm font-bold shrink-0">
                            {String.fromCharCode(65 + idx)}
                          </div>
                          <div className="text-base text-slate-100 leading-relaxed">{opt}</div>
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

                  {/* Actions */}
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
                      <button className="btn-pixel px-6 py-3 rounded-xl" onClick={next}>
                        {qNum >= TOTAL ? "See Results" : "Next →"}
                      </button>
                    )}
                  </div>

                  {/* Explanation */}
                  {showExplain && (
                    <div className="mt-6 rounded-2xl border-2 border-[rgba(56,189,248,0.35)] bg-[rgba(56,189,248,0.05)] p-5">
                      <div className="flex items-center gap-2">
                        <span className="font-bold">
                          {selected === (mcq?.correctIndex ?? mcq?.correct_index ?? 0) ? "✅ Correct!" : "❌ Wrong"}
                        </span>
                        <span className="text-sm text-slate-300/80">
                          Answer: {String.fromCharCode(65 + (mcq?.correctIndex ?? mcq?.correct_index ?? 0))}
                        </span>
                      </div>
                      <div className="mt-2 text-slate-200/90 leading-relaxed">
                        {mcq.explanation || ""}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Quick stats bar */}
        <div className="mt-6 flex gap-3 flex-wrap">
          <div className="px-4 py-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] text-sm">
            Score: {totalCorrect}/{qNum}
          </div>
          <div className="px-4 py-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] text-sm">
            Streak: {streak} 🔥
          </div>
          <div className="px-4 py-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] text-sm">
            XP: {totalCorrect * 20}
          </div>
        </div>
      </div>
    </div>
  );
}
