import { useState } from "react";
import { chapters, characters } from "../data/chapters";
import bg from "../assets/bg.gif";
import PixelFooter from "../components/PixelFooter";

export default function HomePage({ selectedCharacter, onSelectCharacter, onSelectChapter, onDashboard }) {
  const [charPick, setCharPick] = useState(selectedCharacter?.id || null);

  const handleStart = (chapter) => {
    if (!charPick) return;
    const char = characters.find((c) => c.id === charPick);
    onSelectCharacter(char);
    onSelectChapter(chapter);
  };

  return (
    <div className="min-h-screen text-slate-100">
      {/* Hero */}
      <header className="relative min-h-[90vh]">
        <div className="hero-bg-cover relative -z-20">
          <img src={bg} alt="background" className="block w-full h-auto" />
        </div>
        <div className="absolute inset-0 -z-10 bg-[linear-gradient(180deg,rgba(15,23,42,0.35)_0%,rgba(30,41,59,0.45)_50%,rgba(15,23,42,0.85)_100%)]" />
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(900px_420px_at_50%_20%,rgba(94,234,212,0.12),transparent_55%)]" />

        <div className="absolute top-0 left-0 right-0 min-h-[90vh] flex items-center z-10">
          <div className="relative max-w-6xl mx-auto px-6 w-full">
            <div className="max-w-2xl">
              <div className="text-xs tracking-[0.35em] text-slate-200/80">
                NEUROLEARN • ADAPTIVE LEARNING
              </div>
              <div className="pixel-heading mt-6 text-4xl md:text-6xl leading-tight">
                Learn Math Like a Game
              </div>
              <p className="mt-6 text-base md:text-lg text-slate-200/90 leading-relaxed">
                Story-based quests with your favourite characters. Adaptive difficulty powered by AI.
              </p>
              <div className="mt-10 flex items-center gap-4 flex-wrap">
                <button
                  onClick={() => document.getElementById("characters")?.scrollIntoView({ behavior: "smooth" })}
                  className="px-7 py-3 rounded-xl bg-[rgb(94,234,212)] text-slate-900 font-semibold hover:brightness-110 transition shadow-lg"
                >
                  Choose Your Hero
                </button>
                <button
                  onClick={onDashboard}
                  className="px-7 py-3 rounded-xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.85)] transition"
                >
                  Parent Dashboard
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10">
        {/* Character Selection */}
        <section id="characters" className="mt-2">
          <div className="pixel-heading text-2xl md:text-3xl">Choose Your Hero</div>
          <p className="mt-2 text-slate-300/80">Pick a character to guide you through the adventure</p>
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-5">
            {characters.map((c) => (
              <button
                key={c.id}
                onClick={() => setCharPick(c.id)}
                className={`relative text-left rounded-3xl border-2 p-6 transition overflow-hidden shadow-lg
                  ${charPick === c.id
                    ? "border-[rgba(94,234,212,0.85)] bg-[rgba(94,234,212,0.08)] ring-2 ring-[rgba(94,234,212,0.3)]"
                    : "border-[rgba(51,65,85,0.85)] bg-[rgba(15,23,42,0.72)] hover:bg-[rgba(30,41,59,0.85)]"
                  }`}
              >
                <div className="flex items-center gap-4">
                  <img src={c.image} alt={c.name} className="w-20 h-20 object-contain rounded-2xl" />
                  <div>
                    <div className="pixel-heading text-lg">{c.name}</div>
                    <div className="mt-1 text-sm text-slate-300/80 capitalize">{c.personality}</div>
                  </div>
                </div>
                {charPick === c.id && (
                  <div className="absolute top-3 right-3 w-6 h-6 rounded-full bg-[rgb(94,234,212)] flex items-center justify-center text-slate-900 text-xs font-bold">
                    ✓
                  </div>
                )}
              </button>
            ))}
          </div>
        </section>

        {/* Chapters */}
        <section className="mt-16">
          <div className="pixel-heading text-2xl md:text-3xl">Chapters</div>
          <p className="mt-2 text-slate-300/80">Select a chapter to start your quest</p>
          <div className="mt-6 grid grid-cols-1 gap-5">
            {chapters.map((ch) => (
              <button
                key={ch.id}
                onClick={() => handleStart(ch)}
                disabled={!charPick}
                className={`text-left rounded-3xl border-2 p-8 transition shadow-lg
                  ${!charPick
                    ? "border-[rgba(51,65,85,0.6)] bg-[rgba(15,23,42,0.6)] opacity-70 cursor-not-allowed"
                    : "border-[rgba(51,65,85,0.85)] bg-[rgba(15,23,42,0.72)] hover:bg-[rgba(30,41,59,0.85)]"
                  }`}
              >
                <div className="flex items-start justify-between gap-4 flex-wrap">
                  <div>
                    <div className="pixel-heading text-xl md:text-2xl">{ch.title}</div>
                    <div className="mt-3 text-slate-200/90 leading-relaxed">{ch.description}</div>
                    <div className="mt-4 flex gap-2 flex-wrap">
                      <span className="text-xs px-3 py-1 rounded-full border border-[rgba(94,234,212,0.35)] bg-[rgba(94,234,212,0.10)]">
                        {ch.sections.length} Sections
                      </span>
                      <span className="text-xs px-3 py-1 rounded-full border border-[rgba(56,189,248,0.35)] bg-[rgba(56,189,248,0.10)]">
                        Final Boss
                      </span>
                      <span className="text-xs px-3 py-1 rounded-full border border-[rgba(250,204,21,0.35)] bg-[rgba(250,204,21,0.10)]">
                        Adaptive AI
                      </span>
                    </div>
                  </div>
                  <div className={`text-sm px-4 py-2 rounded-full border ${
                    charPick
                      ? "border-[rgba(94,234,212,0.55)] bg-[rgba(94,234,212,0.10)]"
                      : "border-[rgba(51,65,85,0.7)] bg-[rgba(30,41,59,0.4)]"
                  }`}>
                    {charPick ? "Start →" : "Pick a hero first"}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </section>
      </main>

      <PixelFooter />
    </div>
  );
}
