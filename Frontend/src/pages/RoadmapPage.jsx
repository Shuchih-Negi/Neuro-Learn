import { useState } from "react";

export default function RoadmapPage({ chapter, character, sectionProgress, onLearn, onTest, onFinalBoss, onBack }) {
  const [modal, setModal] = useState(null); // section object or null

  const isSectionDone = (sec) => {
    const p = sectionProgress[sec.id];
    return p && p.answered >= 10 && p.correct >= 7;
  };

  const isSectionUnlocked = (idx) => {
    if (idx === 0) return true;
    return isSectionDone(chapter.sections[idx - 1]);
  };

  const allDone = chapter.sections.every(isSectionDone);

  return (
    <div className="min-h-screen text-slate-100">
      {/* Top bar */}
      <div className="sticky top-0 z-50 backdrop-blur-md border-b border-white/[0.06] bg-[rgba(15,23,42,0.82)]">
        <div className="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
          <button onClick={onBack} className="px-4 py-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.85)] transition text-sm">
            ← Home
          </button>
          <div className="flex items-center gap-3">
            <img src={character.image} alt={character.name} className="w-8 h-8 object-contain rounded-lg" />
            <span className="text-sm font-semibold">{character.name}</span>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-10">
        <div className="pixel-heading text-2xl md:text-3xl">{chapter.title}</div>
        <p className="mt-2 text-slate-300/80">Follow the roadmap to master this chapter</p>

        {/* Roadmap */}
        <div className="mt-10 relative">
          {/* Vertical line */}
          <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-[rgba(51,65,85,0.6)]" />

          {chapter.sections.map((sec, idx) => {
            const done = isSectionDone(sec);
            const unlocked = isSectionUnlocked(idx);
            const progress = sectionProgress[sec.id];
            const answered = progress?.answered || 0;
            const correct = progress?.correct || 0;

            return (
              <div key={sec.id} className="relative pl-16 pb-10">
                {/* Node */}
                <div className={`absolute left-4 w-5 h-5 rounded-full border-2 top-1 ${
                  done
                    ? "bg-[rgb(52,211,153)] border-[rgb(52,211,153)]"
                    : unlocked
                    ? "bg-[rgb(56,189,248)] border-[rgb(56,189,248)] animate-pulse"
                    : "bg-[rgba(51,65,85,0.6)] border-[rgba(51,65,85,0.6)]"
                }`} />

                <button
                  onClick={() => unlocked && setModal(sec)}
                  disabled={!unlocked}
                  className={`w-full text-left rounded-2xl border-2 p-5 transition ${
                    !unlocked
                      ? "border-[rgba(51,65,85,0.5)] bg-[rgba(15,23,42,0.5)] opacity-60 cursor-not-allowed"
                      : done
                      ? "border-[rgba(52,211,153,0.5)] bg-[rgba(52,211,153,0.05)] hover:bg-[rgba(52,211,153,0.1)]"
                      : "border-[rgba(56,189,248,0.4)] bg-[rgba(15,23,42,0.72)] hover:bg-[rgba(30,41,59,0.85)]"
                  }`}
                >
                  <div className="flex items-center justify-between gap-3 flex-wrap">
                    <div>
                      <div className="text-xs text-slate-300/70 tracking-wider">SECTION {idx + 1}</div>
                      <div className="pixel-heading text-base mt-1">{sec.title}</div>
                      <div className="mt-1 text-sm text-slate-300/80">{sec.description}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      {done && <span className="text-xs px-3 py-1 rounded-full border border-[rgba(52,211,153,0.55)] bg-[rgba(52,211,153,0.10)]">Completed ✓</span>}
                      {!unlocked && <span className="text-2xl">🔒</span>}
                      {unlocked && !done && <span className="text-xs px-3 py-1 rounded-full border border-[rgba(56,189,248,0.55)] bg-[rgba(56,189,248,0.10)]">Ready</span>}
                    </div>
                  </div>
                  {answered > 0 && (
                    <div className="mt-3">
                      <div className="flex items-center justify-between text-xs text-slate-300/70">
                        <span>Score: {correct}/{answered}</span>
                      </div>
                      <div className="mt-1 h-1.5 w-full rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
                        <div className="h-1.5 bg-gradient-to-r from-[rgb(56,189,248)] to-[rgb(94,234,212)] transition-all" style={{ width: `${Math.round((correct / Math.max(1, answered)) * 100)}%` }} />
                      </div>
                    </div>
                  )}
                </button>
              </div>
            );
          })}

          {/* Final Boss */}
          <div className="relative pl-16 pb-4">
            <div className={`absolute left-4 w-5 h-5 rounded-full border-2 top-1 ${
              allDone ? "bg-[rgb(250,204,21)] border-[rgb(250,204,21)] animate-pulse" : "bg-[rgba(51,65,85,0.6)] border-[rgba(51,65,85,0.6)]"
            }`} />
            <button
              onClick={() => allDone && onFinalBoss()}
              disabled={!allDone}
              className={`w-full text-left rounded-2xl border-2 p-5 transition ${
                allDone
                  ? "border-[rgba(250,204,21,0.6)] bg-[rgba(250,204,21,0.05)] hover:bg-[rgba(250,204,21,0.1)]"
                  : "border-[rgba(51,65,85,0.5)] bg-[rgba(15,23,42,0.5)] opacity-60 cursor-not-allowed"
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-xs text-slate-300/70 tracking-wider">FINAL CHALLENGE</div>
                  <div className="pixel-heading text-base mt-1">{chapter.finalBoss.title}</div>
                  <div className="mt-1 text-sm text-slate-300/80">{chapter.finalBoss.description}</div>
                </div>
                {!allDone ? <span className="text-2xl">🔒</span> : <span className="text-2xl">⚔️</span>}
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Learn / Test Modal */}
      {modal && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/50" onClick={() => setModal(null)} />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="w-full max-w-md rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgb(10,20,44)] shadow-2xl overflow-hidden">
              <div className="p-6 border-b border-[rgba(48,68,105,0.6)] bg-[rgba(13,26,58,0.55)]">
                <div className="pixel-heading text-lg">{modal.title}</div>
                <div className="mt-2 text-sm text-slate-300/80">{modal.description}</div>
              </div>
              <div className="p-6 space-y-3">
                <button
                  onClick={() => { setModal(null); onLearn(modal); }}
                  className="w-full px-6 py-4 rounded-xl border-2 border-[rgba(94,234,212,0.5)] bg-[rgba(94,234,212,0.08)] hover:bg-[rgba(94,234,212,0.15)] transition text-left"
                >
                  <div className="flex items-center gap-4">
                    <span className="text-3xl">📖</span>
                    <div>
                      <div className="font-bold text-lg">Learn</div>
                      <div className="text-sm text-slate-300/80">{character.name} teaches you through a story quest</div>
                    </div>
                  </div>
                </button>
                <button
                  onClick={() => { setModal(null); onTest(modal); }}
                  className="w-full px-6 py-4 rounded-xl border-2 border-[rgba(56,189,248,0.5)] bg-[rgba(56,189,248,0.08)] hover:bg-[rgba(56,189,248,0.15)] transition text-left"
                >
                  <div className="flex items-center gap-4">
                    <span className="text-3xl">⚔️</span>
                    <div>
                      <div className="font-bold text-lg">Test</div>
                      <div className="text-sm text-slate-300/80">10 adaptive questions — prove your skills!</div>
                    </div>
                  </div>
                </button>
              </div>
              <div className="p-4 border-t border-[rgba(48,68,105,0.6)] flex justify-end">
                <button onClick={() => setModal(null)} className="px-4 py-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.8)] text-sm">
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
