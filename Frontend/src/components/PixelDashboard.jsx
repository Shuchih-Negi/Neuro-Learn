// src/components/PixelDashboard.jsx
import { useEffect, useMemo, useState } from "react";
import { moduleBank } from "../modules/moduleBank";
import PixelFooter from "./PixelFooter";
import {
  applyUnlockRules,
  computeMastery,
  loadProgress,
  saveProgress,
} from "../utils/mastery";

// Put your pixel background image inside: src/assets/bg.gif
import bg from "../assets/bg.gif";

function tag(topic) {
  const t = String(topic || "").toLowerCase();
  if (t.includes("fraction")) return "Fractions";
  if (t.includes("algebra")) return "Algebra";
  if (t.includes("linear")) return "Equations";
  if (t.includes("geometry")) return "Geometry";
  if (t.includes("stat") || t.includes("prob")) return "Stats";
  return "Quests";
}

function ProgressBar({ pct }) {
  return (
    <div className="mt-3 h-2 w-full rounded-full bg-[rgba(255,255,255,0.10)] overflow-hidden">
      <div
        className="h-2 bg-gradient-to-r from-[rgb(94,234,212)] to-[rgb(56,189,248)] transition-all"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export default function PixelDashboard({ onSelect }) {
  const [progress, setProgress] = useState(() => loadProgress(moduleBank));

  useEffect(() => {
    const updated = applyUnlockRules(progress, moduleBank);
    setProgress(updated);
    saveProgress(updated);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const computed = useMemo(() => {
    const map = {};
    moduleBank.forEach((m) => {
      const p = progress[m.id];
      const pct = p?.total ? Math.round((p.answered / p.total) * 100) : 0;
      const mastery = p ? computeMastery(p) : { mastered: false, score: 0, reasons: [] };
      map[m.id] = { ...p, pct, mastery };
    });
    return map;
  }, [progress]);

  const handleContinue = () => {
    const first = moduleBank.find((m) => progress[m.id]?.unlocked);
    if (first) onSelect(first);
  };

  return (
    <div className="min-h-screen text-slate-100">
      <header className="relative min-h-[90vh]">

            {/* Background GIF - full width, full height (no crop), hero grows with image */}
            <div className="hero-bg-cover relative -z-20">
                <img src={bg} alt="background" className="block w-full h-auto" />
            </div>

            {/* Dark gradient overlay */}
            <div className="absolute inset-0 -z-10 bg-[linear-gradient(180deg,rgba(15,23,42,0.35)_0%,rgba(30,41,59,0.45)_50%,rgba(15,23,42,0.85)_100%)]" />

            {/* Soft teal glow */}
            <div className="absolute inset-0 -z-10 bg-[radial-gradient(900px_420px_at_50%_20%,rgba(94,234,212,0.12),transparent_55%)]" />

            {/* Content - fixed to first viewport so it stays visible above overlay */}
            <div className="absolute top-0 left-0 right-0 min-h-[90vh] flex items-center z-10">
            <div className="relative max-w-6xl mx-auto px-6 w-full">
                <div className="max-w-2xl">

                <div className="text-xs tracking-[0.35em] text-slate-200/80">
                    STORY QUESTS • ADHD FRIENDLY
                </div>

                <div className="pixel-heading mt-6 text-4xl md:text-6xl leading-tight">
                    Learn Math Like a Game
                </div>

                <p className="mt-6 text-base md:text-lg text-slate-200/90 leading-relaxed">
                    Story-based MCQs with attention modes (Focused / Drifting / Impulsive /
                    Overwhelmed) + adaptive difficulty.
                </p>

                <div className="mt-10 flex items-center gap-4 flex-wrap">
                    <button
                    onClick={handleContinue}
                    className="px-7 py-3 rounded-xl bg-[rgb(94,234,212)] text-slate-900 font-semibold hover:brightness-110 transition shadow-lg"
                    >
                    Start Learning
                    </button>

                    <button
                    onClick={() => document.getElementById("quests")?.scrollIntoView({ behavior: "smooth" })}
                    className="px-7 py-3 rounded-xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.85)] transition"
                    >
                    Explore Modules
                    </button>
                </div>

                <div className="mt-10 mb-2 text-sm text-slate-300/80">
                    No credit card required • Demo build • Fast & clean
                </div>

                </div>
            </div>
            </div>
        </header>

      {/* CONTENT */}
      <main className="max-w-6xl mx-auto px-6 py-10">
        {/* MODULES */}
        <section id="quests" className="mt-2">
          <div className="pixel-heading text-2xl md:text-3xl">Explore Quests</div>
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-5">
            {moduleBank.map((m, idx) => {
              const p = computed[m.id];
              const locked = !progress[m.id]?.unlocked;
              const mastered = Boolean(progress[m.id]?.mastered);

              return (
                <button
                  key={m.id}
                  onClick={() => !locked && onSelect(m)}
                  disabled={locked}
                  className={`text-left rounded-3xl border-2 p-6 transition relative overflow-hidden shadow-[0_6px_20px_rgba(255,255,255,0.05),0_12px_28px_rgba(148,163,184,0.08)]
                    ${
                      locked
                        ? "border-[rgba(51,65,85,0.6)] bg-[rgba(15,23,42,0.6)] opacity-90 cursor-not-allowed"
                        : "border-[rgba(51,65,85,0.85)] bg-[rgba(15,23,42,0.72)] hover:bg-[rgba(30,41,59,0.85)]"
                    }`}
                >
                  {locked && (
                    <div className="absolute right-5 top-5 text-4xl pointer-events-none" aria-hidden>🔒</div>
                  )}

                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="pixel-heading text-lg">
                        {idx + 1}. {m.title}
                      </div>
                      <div className="mt-2 text-sm text-slate-300/80 leading-relaxed break-words">
                        {m.description}
                      </div>
                    </div>
                  </div>

                  <div className="mt-4">
                    <div className="flex items-center justify-between text-xs text-slate-300/80">
                      <span>
                        Progress: {p?.answered || 0}/{p?.total || 10}
                      </span>
                      <span>
                        Mastery:{" "}
                        <span className="text-[rgb(94,234,212)] font-semibold">
                          {p?.mastery?.score ?? 0}%
                        </span>
                      </span>
                    </div>
                    <ProgressBar pct={p?.pct || 0} />
                  </div>

                  <div className="mt-4 flex gap-2 flex-wrap">
                    <span className="text-xs px-3 py-1 rounded-full border border-[rgba(94,234,212,0.35)] bg-[rgba(94,234,212,0.10)]">
                      {tag(m.geminiTopic)}
                    </span>
                    <span className="text-xs px-3 py-1 rounded-full border border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)]">
                      Age {m.age}
                    </span>

                    {mastered && (
                      <span className="text-xs px-3 py-1 rounded-full border border-[rgba(52,211,153,0.55)] bg-[rgba(52,211,153,0.10)]">
                        Mastered
                      </span>
                    )}
                  </div>

                  <div className="mt-5 flex items-center justify-between">
                    <div className="text-xs text-slate-300/70">
                      Fixed set • Adaptive • Attention-aware
                    </div>
                    <div
                      className={`text-xs px-3 py-1 rounded-full border ${
                        locked
                          ? "border-[rgba(51,65,85,0.7)] bg-[rgba(30,41,59,0.4)]"
                          : "border-[rgba(94,234,212,0.55)] bg-[rgba(94,234,212,0.10)]"
                      }`}
                    >
                      {locked ? "Locked" : "Start →"}
                    </div>
                  </div>

                  {locked && (
                    <p className="mt-4 mb-1 text-xs text-slate-400/90">
                      Finish previous module to unlock
                    </p>
                  )}
                </button>
              );
            })}
          </div>
        </section>
      </main>

      <PixelFooter />

      {/* subtle star overlay */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute inset-0 opacity-[0.18] animate-pulse">
          <div className="absolute left-[12%] top-[20%] h-1 w-1 bg-white rounded" />
          <div className="absolute left-[40%] top-[12%] h-1 w-1 bg-white rounded" />
          <div className="absolute left-[72%] top-[24%] h-1 w-1 bg-white rounded" />
          <div className="absolute left-[85%] top-[10%] h-1 w-1 bg-white rounded" />
          <div className="absolute left-[20%] top-[72%] h-1 w-1 bg-white rounded" />
          <div className="absolute left-[58%] top-[78%] h-1 w-1 bg-white rounded" />
        </div>
      </div>
    </div>
  );
}