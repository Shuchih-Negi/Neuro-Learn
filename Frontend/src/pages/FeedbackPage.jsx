import { useEffect, useRef, useState } from "react";
import { generateFeedback } from "../utils/api";
import boxImg from "../assets/box.png";

export default function FeedbackPage({ section, character, results, onContinue }) {
  const [loading, setLoading] = useState(true);
  const [feedback, setFeedback] = useState(null);
  const [typed, setTyped] = useState("");
  const [doneTyping, setDoneTyping] = useState(false);
  const [xpShown, setXpShown] = useState(0);
  const timerRef = useRef(null);

  const pct = Math.round((results.totalCorrect / Math.max(1, results.totalQuestions)) * 100);

  useEffect(() => {
    let cancelled = false;
    generateFeedback({
      character: character.name,
      totalCorrect: results.totalCorrect,
      totalQuestions: results.totalQuestions,
      sectionTitle: section.title,
      attentionHistory: results.attentionHistory || [],
    })
      .then((data) => {
        if (!cancelled) {
          setFeedback(data);
          setLoading(false);
          startTypewriter(data.message || "Great job!");
          animateXp(data.xp_earned || results.totalCorrect * 20);
        }
      })
      .catch(() => {
        if (!cancelled) {
          const xp = results.totalCorrect * 20;
          setFeedback({
            message: `You scored ${results.totalCorrect}/${results.totalQuestions}! Keep going!`,
            tip: "Review the questions you got wrong.",
            xp_earned: xp,
            rating: pct >= 70 ? "good" : "keep_trying",
          });
          setLoading(false);
          setDoneTyping(true);
          setTyped(`You scored ${results.totalCorrect}/${results.totalQuestions}! Keep going!`);
          setXpShown(xp);
        }
      });

    return () => {
      cancelled = true;
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startTypewriter = (text) => {
    if (timerRef.current) clearInterval(timerRef.current);
    setTyped("");
    setDoneTyping(false);
    let i = 0;
    timerRef.current = setInterval(() => {
      i += 2;
      setTyped(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(timerRef.current);
        timerRef.current = null;
        setDoneTyping(true);
      }
    }, 22);
  };

  const animateXp = (target) => {
    let current = 0;
    const step = Math.ceil(target / 30);
    const iv = setInterval(() => {
      current = Math.min(current + step, target);
      setXpShown(current);
      if (current >= target) clearInterval(iv);
    }, 40);
  };

  const ratingEmoji = feedback?.rating === "excellent" ? "🏆" : feedback?.rating === "good" ? "⭐" : "💪";

  return (
    <div className="min-h-screen text-slate-100 flex items-center justify-center">
      <div className="max-w-2xl w-full mx-auto px-6 py-10">
        {loading ? (
          <div className="flex flex-col items-center py-20">
            <div className="pixel-heading text-xl">{character.name} is reviewing your results...</div>
            <div className="mt-4 h-2 w-48 rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
              <div className="h-2 w-1/3 bg-[rgb(56,189,248)] animate-pulse" />
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Score card */}
            <div className="rounded-3xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(10,20,44,0.75)] shadow-xl p-8 text-center">
              <div className="text-6xl">{ratingEmoji}</div>
              <div className="pixel-heading text-2xl mt-4">Quest Complete!</div>
              <div className="mt-2 text-slate-300/80">{section.title}</div>

              <div className="mt-6 flex justify-center gap-8">
                <div>
                  <div className="text-3xl font-bold text-[rgb(94,234,212)]">{results.totalCorrect}/{results.totalQuestions}</div>
                  <div className="text-xs text-slate-300/70 mt-1">Score</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-[rgb(250,204,21)]">+{xpShown}</div>
                  <div className="text-xs text-slate-300/70 mt-1">XP Earned</div>
                </div>
                <div>
                  <div className="text-3xl font-bold">{pct}%</div>
                  <div className="text-xs text-slate-300/70 mt-1">Accuracy</div>
                </div>
              </div>

              {/* Progress bar */}
              <div className="mt-6 h-3 w-full rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
                <div
                  className="h-3 bg-gradient-to-r from-[rgb(56,189,248)] to-[rgb(94,234,212)] transition-all duration-1000"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>

            {/* Character feedback */}
            <div className="flex gap-4 items-start">
              <div className="shrink-0 hidden md:block">
                <img src={character.image} alt={character.name} className="w-24 h-24 object-contain rounded-2xl" />
              </div>
              <div
                className="flex-1 relative rounded-2xl border-2 border-[rgba(48,68,105,0.9)] overflow-hidden"
                style={{ backgroundImage: `url(${boxImg})`, backgroundSize: "cover", backgroundPosition: "center" }}
              >
                <div className="absolute inset-0 bg-[rgba(10,20,44,0.88)]" />
                <div className="relative p-6">
                  <div className="md:hidden flex items-center gap-3 mb-3">
                    <img src={character.image} alt={character.name} className="w-10 h-10 object-contain rounded-xl" />
                    <span className="font-semibold">{character.name}</span>
                  </div>
                  <div className="text-base text-slate-100/95 leading-relaxed">
                    {typed}
                    {!doneTyping && <span className="tw-caret" />}
                  </div>
                </div>
              </div>
            </div>

            {/* Tip */}
            {feedback?.tip && doneTyping && (
              <div className="rounded-2xl border-2 border-[rgba(250,204,21,0.35)] bg-[rgba(250,204,21,0.05)] p-5">
                <div className="font-semibold text-[rgb(250,204,21)]">💡 Tip</div>
                <div className="mt-1 text-slate-200/90">{feedback.tip}</div>
              </div>
            )}

            {/* Continue */}
            <div className="text-center">
              <button
                onClick={onContinue}
                className="px-8 py-3 rounded-xl bg-[rgb(94,234,212)] text-slate-900 font-semibold hover:brightness-110 transition shadow-lg"
              >
                Continue to Roadmap →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
