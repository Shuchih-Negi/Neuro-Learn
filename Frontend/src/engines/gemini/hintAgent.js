import { callGemini, parseJSON } from "./client.js";

export async function generateHints({ question, correctLetter, character, attentionState, accuracy }) {
  const prompt = `You are the HINT AGENT in an adaptive math tutoring system.
You produce TWO graduated hints for a student who is stuck.
- Hint 1: very gentle — points toward the approach without giving anything away
- Hint 2: stronger — tells the student what operation to use and with which numbers

STUDENT PROFILE:
- Character they love: ${character}
- Attention state: ${attentionState}
- Running accuracy: ${accuracy}%

QUESTION: ${question}
Correct answer is option ${correctLetter}.

Rules:
- Never reveal the answer directly
- Reference ${character} if it makes the hint more engaging
- Keep each hint to 1 sentence
- For Overwhelmed state: be extra gentle and encouraging in Hint 1

Return ONLY valid JSON:
{"hint_1":"Gentle hint","hint_2":"Stronger hint"}`;

  try {
    const text = await callGemini(prompt, { temperature: 0.4 });
    return parseJSON(text);
  } catch (e) {
    return {
      hint_1: `Think about what operation you need to use here. ${character} believes in you!`,
      hint_2: "Try breaking the problem into smaller steps and solving one part at a time.",
    };
  }
}
