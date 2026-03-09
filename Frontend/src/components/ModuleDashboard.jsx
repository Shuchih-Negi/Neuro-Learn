// src/components/ModuleDashboard.jsx
import { useMemo, useState } from "react";
import { moduleBank } from "../modules/moduleBank";

function normalize(s) {
  return String(s || "").toLowerCase().trim();
}

function tagFromTopic(topic) {
  const t = normalize(topic);
  if (t.includes("fraction")) return { tag: "Fractions", icon: "🍕" };
  if (t.includes("algebra")) return { tag: "Algebra", icon: "🧩" };
  if (t.includes("linear")) return { tag: "Equations", icon: "📈" };
  if (t.includes("inequal")) return { tag: "Inequalities", icon: "⚖️" };
  if (t.includes("ratio")) return { tag: "Ratio", icon: "🧪" };
  if (t.includes("percent")) return { tag: "Percent", icon: "💸" };
  if (t.includes("exponent") || t.includes("power"))
    return { tag: "Exponents", icon: "⚡" };
  if (t.includes("geometry")) return { tag: "Geometry", icon: "📐" };
  if (t.includes("stat") || t.includes("prob"))
    return { tag: "Stats", icon: "📊" };
  return { tag: "Other", icon: "✨" };
}

function uniqueByTag(arr) {
  const map = new Map();
  for (const item of arr) {
    if (!map.has(item.tag)) map.set(item.tag, item);
  }
  return Array.from(map.values());
}

const chipBase =
  "text-xs px-3 py-2 rounded-2xl border transition select-none flex items-center gap-2";
const chipOn =
  "border-sky-300 bg-sky-100 text-slate-900";
const chipOff =
  "border-slate-200 bg-white text-slate-700 hover:bg-slate-50";

export default function ModuleDashboard({ onSelect }) {
  const [query, setQuery] = useState("");
  const [activeTag, setActiveTag] = useState("All");

  const tags = useMemo(() => {
    const derived = moduleBank.map((m) => tagFromTopic(m.geminiTopic));
    const uniq = uniqueByTag(derived);
    return [{ tag: "All", icon: "🌈" }, ...uniq];
  }, []);

  const filtered = useMemo(() => {
    const q = normalize(query);

    return moduleBank.filter((m) => {
      const text = `${m.title} ${m.description} ${m.geminiTopic} ${m.age}`.toLowerCase();
      const matchesQuery = !q || text.includes(q);

      const t = tagFromTopic(m.geminiTopic).tag;
      const matchesTag = activeTag === "All" || activeTag === t;

      return matchesQuery && matchesTag;
    });
  }, [query, activeTag]);

  return (
    <div className="min-h-screen bg-white text-slate-900">
      {/* Sticky SaaS Navbar */}
      <div className="sticky top-0 z-20 bg-white/85 backdrop-blur border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            {/* Logo */}
            <div className="h-10 w-10 rounded-2xl bg-gradient-to-br from-cyan-100 to-sky-100 border border-sky-200 flex items-center justify-center">
              <div className="h-5 w-5 rounded-full bg-sky-500" />
            </div>

            <div>
              <div className="text-lg font-bold leading-tight">NeuroLearn</div>
              <div className="text-xs text-slate-500 -mt-0.5">
                Pick a quest • Story MCQs • Calm learning ✨
              </div>
            </div>
          </div>

          {/* Search */}
          <div className="w-full max-w-[420px]">
            <div className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2 shadow-sm">
              <span className="text-lg">🔎</span>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search: fractions, algebra, geometry…"
                className="w-full outline-none text-sm placeholder:text-slate-400"
              />
              <button
                onClick={() => setQuery("")}
                className="text-xs px-3 py-1.5 rounded-xl border border-slate-200 hover:bg-slate-50"
              >
                Clear
              </button>
            </div>
          </div>

          {/* Small pill */}
          <div className="hidden md:flex items-center gap-2">
            <span className="text-xs px-3 py-1 rounded-full border border-slate-200 bg-slate-50">
              Age 11–15
            </span>
            <span className="text-xs px-3 py-1 rounded-full border border-slate-200 bg-slate-50">
              Kids-friendly UI
            </span>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Hero */}
        <div className="rounded-[28px] border border-slate-200 bg-gradient-to-br from-cyan-50 to-white p-6 md:p-8 shadow-sm">
          <div className="flex items-start justify-between gap-6 flex-wrap">
            <div>
              <div className="text-3xl font-bold leading-tight">
                Choose your learning quest 🧠✨
              </div>
              <div className="mt-2 text-slate-700 max-w-2xl">
                Story-based MCQs that adapt to your attention. Quick feedback,
                calm UI, and step-by-step solutions when needed.
              </div>

              <div className="mt-4 flex items-center gap-2 flex-wrap">
                <span className="text-xs px-3 py-1 rounded-full border border-sky-200 bg-sky-50">
                  📖 Story questions
                </span>
                <span className="text-xs px-3 py-1 rounded-full border border-sky-200 bg-sky-50">
                  ⚡ Adaptive difficulty
                </span>
                <span className="text-xs px-3 py-1 rounded-full border border-sky-200 bg-sky-50">
                  🧩 Explainable modes
                </span>
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm w-full md:w-[320px]">
              <div className="text-xs text-slate-500">Modules found</div>
              <div className="mt-1 text-3xl font-bold">{filtered.length}</div>
              <div className="mt-2 text-sm text-slate-600">
                Tip: filter by topic below 👇
              </div>
              <div className="mt-3 h-2 w-full rounded-full bg-slate-100 overflow-hidden">
                <div
                  className="h-2 bg-gradient-to-r from-sky-500 to-cyan-400"
                  style={{ width: `${Math.min(100, filtered.length * 12)}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Topic chips */}
        <div className="mt-6 flex items-center gap-2 flex-wrap">
          {tags.map((t) => {
            const on = activeTag === t.tag;
            return (
              <button
                key={t.tag}
                onClick={() => setActiveTag(t.tag)}
                className={`${chipBase} ${on ? chipOn : chipOff}`}
                title={`Filter by ${t.tag}`}
              >
                <span className="text-base">{t.icon}</span>
                <span>{t.tag}</span>
              </button>
            );
          })}
        </div>

        {/* Module cards */}
        <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((m) => {
            const cat = tagFromTopic(m.geminiTopic);

            return (
              <button
                key={m.id}
                onClick={() => onSelect(m)}
                className="group text-left rounded-[28px] border border-slate-200 bg-white p-6 shadow-sm hover:shadow-md hover:border-slate-300 transition"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-lg font-semibold">{m.title}</div>
                    <div className="mt-1 text-sm text-slate-600">
                      {m.description}
                    </div>
                  </div>

                  <div className="text-3xl">{m.emoji}</div>
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  <span className="text-xs px-3 py-1 rounded-full border border-sky-200 bg-sky-50">
                    {cat.icon} <span className="font-semibold">{cat.tag}</span>
                  </span>
                  <span className="text-xs px-3 py-1 rounded-full border border-slate-200 bg-white">
                    Age: <span className="font-semibold">{m.age}</span>
                  </span>
                </div>

                <div className="mt-5 rounded-2xl border border-slate-200 bg-gradient-to-br from-cyan-50 to-white p-3">
                  <div className="text-xs text-slate-500">What you’ll practice</div>
                  <div className="text-sm text-slate-700 mt-1">
                    {m.geminiTopic}
                  </div>
                </div>

                <div className="mt-5 flex items-center justify-between">
                  <div className="text-xs text-slate-500">
                    Fun MCQs • Instant feedback
                  </div>
                  <div className="text-xs px-3 py-1 rounded-full border border-sky-200 bg-sky-50 group-hover:bg-sky-100 transition">
                    Start Quest →
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Bottom helper */}
        <div className="mt-8 rounded-[28px] border border-slate-200 bg-white p-6 shadow-sm">
          <div className="text-sm font-semibold">How NeuroLearn adapts</div>
          <div className="mt-2 text-sm text-slate-700 leading-relaxed">
            We track response time, idle spikes, and retries to infer attention states:
            <span className="font-semibold"> Focused</span>,{" "}
            <span className="font-semibold"> Drifting</span>,{" "}
            <span className="font-semibold"> Impulsive</span>,{" "}
            <span className="font-semibold"> Overwhelmed</span>.
            Then difficulty + support changes automatically.
          </div>
        </div>
      </div>
    </div>
  );
}