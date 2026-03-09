import { generateQuestion } from "./questionAgent.js";
import { wrapInStory } from "./storyAgent.js";
import { validateQuestion } from "./qaAgent.js";
import { generateHints } from "./hintAgent.js";

const MAX_RETRIES = 2;

// Local fallback generator
function localFallback(topic, difficulty, character) {
  const a = Math.floor(Math.random() * 8) + 2;
  const x = Math.floor(Math.random() * 10) + 1;
  const b = Math.floor(Math.random() * 10) + 1;
  const c = a * x + b;
  const correct = `x = ${x}`;
  const wrongs = [`x = ${x + 1}`, `x = ${x - 1}`, `x = ${x + 2}`];
  const all = [correct, ...wrongs].sort(() => Math.random() - 0.5);
  const ci = all.indexOf(correct);

  return {
    question: `${character} needs to solve: ${a}x + ${b} = ${c}. What is x?`,
    options: all.map((o, i) => `${String.fromCharCode(65 + i)}) ${o}`),
    correct_index: ci,
    explanation: `${a}x = ${c} − ${b} = ${a * x}, so x = ${x}.`,
    difficulty,
    hints: {
      hint_1: "Try isolating x by moving the constant to the other side.",
      hint_2: `Subtract ${b} from both sides, then divide by ${a}.`,
    },
  };
}

/**
 * Full multi-agent pipeline:
 * QuestionAgent → StoryAgent → QAAgent (with retries) → HintAgent
 */
export async function generateQuestionPacket({
  topic,
  difficulty,
  attentionState,
  character,
  qNum,
  total,
  accuracy,
  history,
}) {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      // Step 1: QuestionAgent
      const rawQ = await generateQuestion({
        topic, difficulty, attentionState, qNum, total, accuracy, history,
      });
      if (!rawQ?.question_stem) throw new Error("QuestionAgent failed");

      // Step 2: StoryAgent
      const story = await wrapInStory({
        questionData: rawQ, character, topic, attentionState, difficulty,
      });
      if (!story?.question || !story?.options?.length) throw new Error("StoryAgent failed");

      // Step 3: QAAgent
      const qa = await validateQuestion({
        storyQuestion: story, topic, difficulty, attentionState,
      });

      if (!qa.passed && attempt < MAX_RETRIES) {
        continue; // Retry
      }

      // Step 4: HintAgent (runs in parallel — no dependency on QA pass)
      const correctLetter = String.fromCharCode(65 + (story.correct_index ?? 0));
      const hints = await generateHints({
        question: story.question, correctLetter, character, attentionState, accuracy,
      });

      return {
        question: story.question,
        options: story.options,
        correctIndex: story.correct_index ?? 0,
        explanation: story.explanation || "",
        difficulty: story.difficulty ?? difficulty,
        hints,
        agentLog: { rawQ, story, qa, hints, attempt },
      };
    } catch (e) {
      if (attempt === MAX_RETRIES) break;
    }
  }

  // Fallback
  const fb = localFallback(topic, difficulty, character);
  return {
    question: fb.question,
    options: fb.options,
    correctIndex: fb.correct_index,
    explanation: fb.explanation,
    difficulty: fb.difficulty,
    hints: fb.hints,
    agentLog: { fallback: true },
  };
}

/**
 * Generate a learning story/explanation for a topic section
 */
export async function generateLearningStory({ topic, section, character, difficulty }) {
  const { callGemini } = await import("./client.js");

  const prompt = `You are a fun, engaging math tutor explaining "${section}" (part of ${topic}) to a student aged 11-15.
Their favorite character is ${character}.

Create an exciting adventure story (3-4 paragraphs) that teaches the core concept of "${section}" through ${character}'s adventure.
Make it feel like a quest/mission. Include:
- A scenario where ${character} encounters this math concept
- A clear explanation of the concept woven into the story
- An example problem that ${character} solves step by step
- End with encouragement to try it themselves

Keep it engaging, age-appropriate, and educational. Use simple language.
Difficulty level: ${difficulty}/5.

Return ONLY valid JSON:
{"title":"Story title","story":"The full story with \\n for paragraphs","concept":"One-line summary of the math concept","example":{"problem":"An example problem","solution":"Step by step solution"}}`;

  try {
    const text = await callGemini(prompt, { temperature: 0.9 });
    const { parseJSON } = await import("./client.js");
    return parseJSON(text);
  } catch (e) {
    return {
      title: `${character}'s Math Quest`,
      story: `${character} embarks on a mission to master ${section}! Along the way, they discover that math is like solving puzzles.\n\nLet's learn together and see what ${character} finds out!`,
      concept: section,
      example: { problem: "Try the test to practice!", solution: "You'll learn by doing." },
    };
  }
}
