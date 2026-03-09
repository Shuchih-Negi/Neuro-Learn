import { useEffect, useState } from "react";
import { getDashboard } from "../utils/api";
import { chapters } from "../data/chapters";

const STATE_COLORS = {
  Focused: "bg-[rgb(52,211,153)]",
  Drifting: "bg-[rgb(250,204,21)]",
  Impulsive: "bg-[rgb(232,121,249)]",
  Overwhelmed: "bg-[rgb(251,113,133)]",
};

export default function DashboardPage({ onBack }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDashboard("default")
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  const sectionTitleMap = {};
  chapters.forEach((ch) =>
    ch.sections.forEach((s) => {
      sectionTitleMap[s.id] = s.title;
    })
  );

  return (
    <div className="min-h-screen text-slate-100">
      {/* Top bar */}
      <div className="sticky top-0 z-50 backdrop-blur-md border-b border-white/[0.06] bg-[rgba(15,23,42,0.82)]">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
          <button onClick={onBack} className="px-4 py-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.85)] transition text-sm">
            ← Home
          </button>
          <div className="pixel-heading text-base">Parent / Teacher Dashboard</div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-10">
        {loading ? (
          <div className="flex flex-col items-center py-20">
            <div className="pixel-heading text-xl">Loading dashboard...</div>
            <div className="mt-4 h-2 w-48 rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
              <div className="h-2 w-1/3 bg-[rgb(56,189,248)] animate-pulse" />
            </div>
          </div>
        ) : !data || data.total_answered === 0 ? (
          <div className="text-center py-20">
            <div className="text-4xl mb-4">📊</div>
            <div className="pixel-heading text-xl">No Data Yet</div>
            <p className="mt-2 text-slate-300/80">The student hasn't completed any tests yet. Data will appear here after taking quizzes.</p>
          </div>
        ) : (
          <>
            {/* Overview cards */}
            <div className="pixel-heading text-2xl">Student Overview</div>
            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Total XP" value={data.total_xp} color="text-[rgb(250,204,21)]" />
              <StatCard label="Level" value={data.level} color="text-[rgb(56,189,248)]" />
              <StatCard label="Questions" value={`${data.total_correct}/${data.total_answered}`} color="text-[rgb(94,234,212)]" />
              <StatCard label="Accuracy" value={`${data.overall_accuracy}%`} color="text-white" />
            </div>

            {/* Section breakdown */}
            <div className="mt-10">
              <div className="pixel-heading text-xl">Section Performance</div>
              <div className="mt-4 space-y-4">
                {data.sections.map((sec) => (
                  <div key={sec.section_id} className="rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(15,23,42,0.72)] p-5">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div>
                        <div className="font-semibold">{sectionTitleMap[sec.section_id] || sec.section_id}</div>
                        <div className="text-sm text-slate-300/80 mt-1">
                          {sec.correct}/{sec.answered} correct • Best streak: {sec.best_streak}
                        </div>
                      </div>
                      <div className="text-2xl font-bold text-[rgb(94,234,212)]">{sec.accuracy}%</div>
                    </div>

                    {/* Accuracy bar */}
                    <div className="mt-3 h-2 w-full rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
                      <div className="h-2 bg-gradient-to-r from-[rgb(56,189,248)] to-[rgb(94,234,212)] transition-all" style={{ width: `${sec.accuracy}%` }} />
                    </div>

                    {/* Attention state breakdown */}
                    {sec.state_counts && Object.keys(sec.state_counts).length > 0 && (
                      <div className="mt-3 flex gap-2 flex-wrap">
                        {Object.entries(sec.state_counts).map(([state, count]) => (
                          <span key={state} className="flex items-center gap-1.5 text-xs text-slate-300/80">
                            <span className={`w-2.5 h-2.5 rounded-full ${STATE_COLORS[state] || "bg-slate-500"}`} />
                            {state}: {count}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Activity */}
            {data.recent_history?.length > 0 && (
              <div className="mt-10">
                <div className="pixel-heading text-xl">Recent Activity</div>
                <div className="mt-4 rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(15,23,42,0.72)] p-5">
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {data.recent_history.slice().reverse().map((h, i) => (
                      <div key={i} className="flex items-center gap-3 text-sm">
                        <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${h.correct ? "bg-[rgb(52,211,153)]" : "bg-[rgb(251,113,133)]"}`} />
                        <span className="text-slate-300/80">{sectionTitleMap[h.section_id] || h.section_id}</span>
                        <span className="text-slate-400/60">•</span>
                        <span className={`${STATE_COLORS[h.state] ? "text-slate-200/90" : ""}`}>{h.state}</span>
                        <span className="text-slate-400/60">•</span>
                        <span className="text-slate-400/70">{h.rt?.toFixed(1)}s</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Attention Legend */}
            <div className="mt-10 rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(15,23,42,0.72)] p-5">
              <div className="pixel-heading text-base">Attention States Guide</div>
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-slate-200/90">
                <div className="flex items-start gap-2">
                  <span className="w-3 h-3 rounded-full bg-[rgb(52,211,153)] mt-0.5 shrink-0" />
                  <span><strong>Focused</strong> — engaged and answering correctly</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="w-3 h-3 rounded-full bg-[rgb(250,204,21)] mt-0.5 shrink-0" />
                  <span><strong>Drifting</strong> — slow responses, attention wandering</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="w-3 h-3 rounded-full bg-[rgb(232,121,249)] mt-0.5 shrink-0" />
                  <span><strong>Impulsive</strong> — fast but incorrect answers</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="w-3 h-3 rounded-full bg-[rgb(251,113,133)] mt-0.5 shrink-0" />
                  <span><strong>Overwhelmed</strong> — struggling, needs simpler problems</span>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div className="rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgba(15,23,42,0.72)] p-4">
      <div className="text-xs text-slate-300/70">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${color}`}>{value}</div>
    </div>
  );
}
