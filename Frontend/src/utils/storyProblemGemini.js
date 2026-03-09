// src/utils/storyProblemGemini.js
import { ai } from "./gemini";

// Strict JSON-only generator (easy to parse)
export async function generateStoryProblem({
  topic,
  ageGroup = "11-15",
  difficulty = 2, // 1–5
  theme = "school-life in India",
}) {
  const prompt = `
You are a math problem writer for learners age ${ageGroup}.
Create ONE short, engaging STORY word problem based on topic: "${topic}".
Difficulty: ${difficulty} on a 1–5 scale.

Rules:
- The story must be realistic and relatable: ${theme}.
- Must have exactly ONE final numeric answer.
- Use clean numbers (avoid ugly decimals unless topic requires it).
- Provide a small hint and 3-6 step solution.
- Output MUST be valid JSON ONLY (no markdown, no extra text).

JSON schema:
{
  "question": "string",
  "answer": number,
  "unit": "string or empty",
  "hint": "string",
  "steps": ["string", "..."],
  "difficulty": number,
  "topic": "string"
}
`;

  const res = await ai.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: prompt,
  });

  const text = res.text ?? "";

  // Parse safely (Gemini might occasionally add whitespace/newlines)
  try {
    const jsonStart = text.indexOf("{");
    const jsonEnd = text.lastIndexOf("}");
    const jsonStr = text.slice(jsonStart, jsonEnd + 1);
    const data = JSON.parse(jsonStr);

    // Tiny validation fallback
    if (!data.question || typeof data.answer !== "number") {
      throw new Error("Invalid JSON structure");
    }
    return data;
  } catch (e) {
    // Fallback object if parsing fails (rare but possible)
    return {
      question: `Fallback: Solve a ${topic} question (Gemini output parse failed).`,
      answer: 0,
      unit: "",
      hint: "Try again / refresh question.",
      steps: ["Gemini output couldn't be parsed."],
      difficulty,
      topic,
    };
  }
}