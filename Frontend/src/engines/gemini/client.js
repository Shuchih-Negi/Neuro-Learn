import { GoogleGenAI } from "@google/genai";

const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
export const ai = new GoogleGenAI({ apiKey });

const TIMEOUT_MS = 12000;

export function withTimeout(promise, ms = TIMEOUT_MS) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms)),
  ]);
}

export async function callGemini(prompt, { temperature = 0.8 } = {}) {
  const res = await withTimeout(
    ai.models.generateContent({
      model: "gemini-2.5-flash",  // Updated to match backend
      contents: prompt,
      config: { temperature },
    })
  );
  return res?.text ?? "";
}

export function parseJSON(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end < start) throw new Error("No JSON found");
  return JSON.parse(text.slice(start, end + 1));
}
