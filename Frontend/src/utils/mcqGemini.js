// src/utils/mcqGemini.js
import { ai } from "./gemini";

/**
 * Gemini = primary
 * Local generator = fallback
 * Also: module-specific caching + no-repeat history
 */

const store = new Map();
const TARGET_QUEUE = 10;
const BATCH_SIZE = 6;
const TIMEOUT_MS = 9000;

// Toggle this true if you want to see logs
const DEBUG = false;

function withTimeout(promise, ms = TIMEOUT_MS) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms)),
  ]);
}

function normalize(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[^\w\s₹%./()+-]/g, "")
    .trim();
}

function makeKey({ moduleId, topic, ageGroup, difficulty }) {
  return `${moduleId}::${normalize(topic)}::${ageGroup}::${difficulty}`;
}

function bucketFor(key) {
  if (!store.has(key)) store.set(key, { queue: [], inflight: false, seen: new Set() });
  return store.get(key);
}

/* ---------------------------
   ✅ Local fallback generator
---------------------------- */

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}
function pick(arr) {
  return arr[randInt(0, arr.length - 1)];
}
function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
function makeOptions(correct, wrongs) {
  const opts = shuffle([String(correct), ...wrongs.map(String)]);
  return { options: opts, correctIndex: opts.indexOf(String(correct)) };
}
function fractionSimplify(n, d) {
  const gcd = (a, b) => (b ? gcd(b, a % b) : Math.abs(a));
  const g = gcd(n, d);
  return [n / g, d / g];
}

function genFractions(difficulty) {
  const d = pick([4, 5, 6, 8, 10, 12]);
  const a = randInt(1, d - 1);
  const b = randInt(1, d - 1);
  const op = pick(["+", "-"]);
  const num = op === "+" ? a + b : a - b;
  const [sn, sd] = fractionSimplify(num, d);
  const correct = `${sn}/${sd}`;

  const wrongs = [`${a + b}/${d}`, `${Math.abs(a - b)}/${d}`, `${sn}/${sd + 1}`];
  const { options, correctIndex } = makeOptions(correct, wrongs);

  return {
    topic: "fractions",
    difficulty,
    question: `What is ${a}/${d} ${op} ${b}/${d} ?`,
    options,
    correctIndex,
    explanation: `Same denominator ${d}. ${a} ${op} ${b} = ${num}. Simplify → ${sn}/${sd}.`,
  };
}

function genLinear(difficulty) {
  const a = randInt(2, 9);
  const x = randInt(1, 12);
  const b = randInt(1, 12);
  const c = a * x + b;

  const correct = `x = ${x}`;
  const wrongs = [`x = ${x + 1}`, `x = ${x - 1}`, `x = ${x + 2}`];
  const { options, correctIndex } = makeOptions(correct, wrongs);

  return {
    topic: "linear equations",
    difficulty,
    question: `Solve: ${a}x + ${b} = ${c}`,
    options,
    correctIndex,
    explanation: `${a}x = ${c} − ${b} = ${a * x}. So x = ${x}.`,
  };
}

function genPercentages(difficulty) {
  const base = pick([100, 120, 150, 200, 250, 300, 400]);
  const pct = pick([5, 10, 12, 15, 20, 25, 30]);
  const type = pick(["discount", "increase"]);
  const change = (base * pct) / 100;
  const correct = type === "discount" ? base - change : base + change;

  const wrongs = [`₹${base}`, `₹${change}`, `₹${base + change}`];
  const { options, correctIndex } = makeOptions(`₹${correct}`, wrongs);

  return {
    topic: "percentages",
    difficulty,
    question:
      type === "discount"
        ? `A product costs ₹${base}. It has a ${pct}% discount. What is the final price?`
        : `A salary of ₹${base} increases by ${pct}%. What is the new salary?`,
    options,
    correctIndex,
    explanation: `${pct}% of ${base} = ${change}. ${type === "discount" ? "Subtract" : "Add"} → ₹${correct}.`,
  };
}

function genRatios(difficulty) {
  const a = randInt(2, 9);
  const b = randInt(2, 9);
  const total = randInt(20, 80) * (a + b);
  const partA = (total * a) / (a + b);

  const { options, correctIndex } = makeOptions(partA, [partA + 10, partA - 10, total - partA]);

  return {
    topic: "ratios and proportions",
    difficulty,
    question: `In a class, boys:girls = ${a}:${b}. If total students are ${total}, how many boys are there?`,
    options,
    correctIndex,
    explanation: `Total parts = ${a + b}. Boys = (${a}/${a + b}) × ${total} = ${partA}.`,
  };
}

function genGeometry(difficulty) {
  const l = randInt(5, 25);
  const w = randInt(4, 20);
  const type = pick(["area", "perimeter"]);
  const correct = type === "area" ? l * w : 2 * (l + w);

  const wrongs = type === "area" ? [l + w, 2 * (l + w), correct + 10] : [l + w, l * w, correct + 8];
  const { options, correctIndex } = makeOptions(correct, wrongs);

  return {
    topic: "geometry",
    difficulty,
    question:
      type === "area"
        ? `A rectangle has length ${l} cm and width ${w} cm. What is its area?`
        : `A rectangle has length ${l} cm and width ${w} cm. What is its perimeter?`,
    options,
    correctIndex,
    explanation:
      type === "area"
        ? `Area = ${l} × ${w} = ${correct}.`
        : `Perimeter = 2(${l} + ${w}) = ${correct}.`,
  };
}

function genStatistics(difficulty) {
  const arr = [randInt(2, 10), randInt(2, 10), randInt(2, 10), randInt(2, 10), randInt(2, 10)];
  const type = pick(["mean", "median"]);
  const sorted = [...arr].sort((a, b) => a - b);
  const correct =
    type === "mean" ? arr.reduce((s, x) => s + x, 0) / arr.length : sorted[Math.floor(sorted.length / 2)];

  const wrongs = type === "mean" ? [correct + 1, correct - 1, arr.reduce((s, x) => s + x, 0)] : [sorted[1], sorted[3], sorted[2] + 1];
  const { options, correctIndex } = makeOptions(correct, wrongs);

  return {
    topic: "statistics",
    difficulty,
    question: type === "mean" ? `Find the mean of: ${arr.join(", ")}` : `Find the median of: ${arr.join(", ")}`,
    options,
    correctIndex,
    explanation:
      type === "mean"
        ? `Mean = (sum)/${arr.length} = ${arr.reduce((s, x) => s + x, 0)}/${arr.length} = ${correct}.`
        : `Sort: ${sorted.join(", ")} → middle value = ${correct}.`,
  };
}

function genAlgebraBasics(difficulty) {
  const a = randInt(2, 9);
  const b = randInt(1, 6);
  const c = randInt(1, 6);
  const correct = `${a + b - c}x`;

  const wrongs = [`${a + b + c}x`, `${a - b - c}x`, `${a + b - c}`];
  const { options, correctIndex } = makeOptions(correct, wrongs);

  return {
    topic: "algebra basics",
    difficulty,
    question: `Simplify: ${a}x + ${b}x − ${c}x`,
    options,
    correctIndex,
    explanation: `Combine like terms: (${a} + ${b} − ${c})x = ${a + b - c}x.`,
  };
}

function genInequalities(difficulty) {
  const a = randInt(2, 9);
  const x = randInt(2, 10);
  const b = randInt(1, 10);
  const c = a * x + b;
  const correct = `x < ${x}`;

  const wrongs = [`x > ${x}`, `x ≤ ${x}`, `x ≥ ${x}`];
  const { options, correctIndex } = makeOptions(correct, wrongs);

  return {
    topic: "inequalities",
    difficulty,
    question: `Solve: ${a}x + ${b} < ${c}`,
    options,
    correctIndex,
    explanation: `${a}x < ${c} − ${b} = ${a * x} ⇒ x < ${x}.`,
  };
}

function genExponents(difficulty) {
  const a = randInt(2, 6);
  const b = randInt(2, 5);
  const correct = a ** b;

  const wrongs = [a * b, a ** (b - 1), (a + 1) ** b];
  const { options, correctIndex } = makeOptions(correct, wrongs);

  return {
    topic: "exponents and powers",
    difficulty,
    question: `What is ${a}^${b}?`,
    options,
    correctIndex,
    explanation: `${a}^${b} = multiply ${a} by itself ${b} times → ${correct}.`,
  };
}

function localMCQ(topic, difficulty) {
  const t = (topic || "").toLowerCase();
  if (t.includes("fraction")) return genFractions(difficulty);
  if (t.includes("linear")) return genLinear(difficulty);
  if (t.includes("percent")) return genPercentages(difficulty);
  if (t.includes("ratio")) return genRatios(difficulty);
  if (t.includes("geometry")) return genGeometry(difficulty);
  if (t.includes("stat")) return genStatistics(difficulty);
  if (t.includes("algebra basics")) return genAlgebraBasics(difficulty);
  if (t.includes("inequal")) return genInequalities(difficulty);
  if (t.includes("exponent")) return genExponents(difficulty);

  return pick([genLinear, genPercentages, genFractions, genRatios, genGeometry, genStatistics])(difficulty);
}

/* ---------------------------
   ✅ Gemini batch fetch
---------------------------- */

async function fetchBatchGemini({ topic, ageGroup, difficulty, batchSize }) {
  const prompt = `
You are generating STORY-BASED math MCQs for ADHD-friendly learning.
Age: ${ageGroup}
Topic: "${topic}"
Difficulty: ${difficulty} (1-5)

Hard rules:
- Generate ${batchSize} DIFFERENT questions.
- Each question must be a SHORT story situation (school/canteen/shopping/cricket/travel).
- Exactly 4 options, exactly one correct.
- Keep numbers clean (avoid ugly decimals).
- Provide a 1–2 line explanation.
- Output JSON ONLY (no markdown, no extra text).

Schema:
{"items":[{"topic":"string","difficulty":number,"question":"string","options":["a","b","c","d"],"correctIndex":0,"explanation":"string","tags":["story","<subtopic>"]}]}
`;
  const res = await withTimeout(
    ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: prompt,
    }),
    TIMEOUT_MS
  );

  const text = res?.text ?? "";
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end < start) throw new Error("bad-json");
  let data;
  try {
    data = JSON.parse(text.slice(start, end + 1));
  } catch {
    throw new Error("bad-json");
  }
  if (!data.items || !Array.isArray(data.items)) throw new Error("bad-json");
  return data.items;
}

async function refill(key, params) {
  const b = bucketFor(key);
  if (b.inflight) return;
  if (b.queue.length >= TARGET_QUEUE) return;

  b.inflight = true;
  try {
    const items = await fetchBatchGemini({ ...params, batchSize: BATCH_SIZE });

    for (const it of items) {
      if (!it?.question || !Array.isArray(it.options) || it.options.length !== 4) continue;
      if (typeof it.correctIndex !== "number" || it.correctIndex < 0 || it.correctIndex > 3) continue;

      const sig = normalize(it.question) + "||" + it.options.map(normalize).join("|");
      if (b.seen.has(sig)) continue;

      b.seen.add(sig);
      b.queue.push(it);
    }
  } catch (e) {
    if (DEBUG) console.error("Gemini refill failed:", e);
  } finally {
    b.inflight = false;
  }
}

export async function generateMCQ({ moduleId, topic, ageGroup = "11-15", difficulty = 2 }) {
  const safeTopic = topic && topic.trim() ? topic.trim() : "general math";
  const key = makeKey({ moduleId, topic: safeTopic, ageGroup, difficulty });
  const b = bucketFor(key);

  // Background refill
  refill(key, { topic: safeTopic, ageGroup, difficulty });

  // Instant if queue has items
  if (b.queue.length > 0) {
    const next = b.queue.shift();
    refill(key, { topic: safeTopic, ageGroup, difficulty });
    return next;
  }

  // Try one awaited refill for first-time
  await refill(key, { topic: safeTopic, ageGroup, difficulty });
  if (b.queue.length > 0) return b.queue.shift();

  // Fallback to local generator (never blocks)
  return localMCQ(safeTopic, difficulty);
}