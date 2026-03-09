import { callGemini, parseJSON } from "./client.js";

const STATE_TONE = {
  Focused: "adventurous and slightly challenging tone",
  Drifting: "funny, surprising, and very short — grab attention fast",
  Impulsive: "start with 'Read carefully:' then use a slightly tricky setup",
  Overwhelmed: "warm, encouraging, super simple — one sentence max",
};

export async function wrapInStory({ questionData, character, topic, attentionState, difficulty }) {
  const tone = STATE_TONE[attentionState] || "friendly and engaging";

  const prompt = `You are the STORY AGENT in an adaptive math tutoring system.
You receive a raw math problem and wrap it in a short story featuring the student's favourite character.
You do NOT change the math — only add narrative.

CHARACTER: ${character}
TONE: ${tone}
TOPIC: ${topic}

RAW PROBLEM:
  Question stem : ${questionData.question_stem}
  Correct answer: ${questionData.correct_answer}
  Distractors   : ${JSON.stringify(questionData.distractors)}

Rules:
- Keep the story to 1-2 sentences max
- The math numbers must be unchanged
- Arrange correct answer + distractors into 4 options (A, B, C, D) — shuffle so correct is NOT always A
- The "correct_index" (0=A,1=B,2=C,3=D) must match where you placed the correct answer
- All options must look plausible

Return ONLY valid JSON:
{"question":"Full story-wrapped question","options":["A) ...","B) ...","C) ...","D) ..."],"correct_index":0,"explanation":"One sentence explaining why the answer is correct","difficulty":${difficulty}}`;

  try {
    const text = await callGemini(prompt, { temperature: 0.9 });
    return parseJSON(text);
  } catch (e) {
    return null;
  }
}
