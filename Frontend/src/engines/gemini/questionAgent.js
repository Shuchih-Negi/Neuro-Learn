import { callGemini, parseJSON } from "./client.js";

const STATE_GUIDANCE = {
  Focused: "a challenging multi-step problem requiring 2-3 operations",
  Drifting: "a short single-step problem with a surprising or funny number to re-engage",
  Impulsive: "a problem with a deliberate common mistake trap — start with 'Read carefully:'",
  Overwhelmed: "the very first micro-step only — one operation, tiny numbers, super simple",
};

export async function generateQuestion({ topic, difficulty, attentionState, qNum, total, accuracy, history }) {
  const guidance = STATE_GUIDANCE[attentionState] || "a standard problem";
  const avoid = history?.length
    ? `\n\nALREADY ASKED (do not repeat):\n${history.slice(-3).map(q => `- ${q}`).join("\n")}`
    : "";

  const prompt = `You are the QUESTION AGENT in an adaptive math tutoring system.
Your only job: produce a mathematically correct ${topic} problem.

CONSTRAINTS:
- Topic: ${topic}
- Difficulty: ${difficulty}/5
- Attention state: ${attentionState}
- Guidance: Create ${guidance}
- Grade level: 6-10 (ages 11-15)
- Question number: ${qNum} of ${total}
- Running accuracy so far: ${accuracy}%
${avoid}

Produce ONE problem. Return ONLY valid JSON:
{"question_stem":"The bare math problem (no story yet)","correct_answer":"exact correct answer as a string","distractors":["wrong1","wrong2","wrong3"],"difficulty_actual":${difficulty},"operation":"e.g. linear equations"}`;

  try {
    const text = await callGemini(prompt, { temperature: 0.8 });
    return parseJSON(text);
  } catch (e) {
    return null;
  }
}
