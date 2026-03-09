import { callGemini, parseJSON } from "./client.js";

export async function validateQuestion({ storyQuestion, topic, difficulty, attentionState }) {
  const prompt = `You are the QA AGENT in an adaptive math tutoring system.
Your job is strict quality control. Be critical but fair.

CONTEXT:
- Topic: ${topic}
- Target difficulty: ${difficulty}/5
- Attention state: ${attentionState}
- Grade level: 6-10

QUESTION TO VALIDATE:
${JSON.stringify(storyQuestion, null, 2)}

CHECK ALL OF THE FOLLOWING:
1. Is the math correct? (verify correct_index points to the right answer)
2. Are the distractors plausible (not obviously wrong)?
3. Is difficulty appropriate for grade 6-10?
4. Is the story age-appropriate and non-offensive?
5. Does the question match the ${topic} topic?
6. For Overwhelmed state: is it truly simple (one step, small numbers)?
7. For Impulsive state: does it have a deliberate trap?

Return ONLY valid JSON:
{"passed":true,"reason":"Brief reason if failed, empty string if passed","suggested_fix":"How to fix it (null if passed)"}`;

  try {
    const text = await callGemini(prompt, { temperature: 0.1 });
    return parseJSON(text);
  } catch (e) {
    return { passed: true, reason: "", suggested_fix: null };
  }
}
