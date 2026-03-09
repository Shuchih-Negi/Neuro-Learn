const BASE = "http://localhost:8000/api";

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${path} failed: ${err}`);
  }
  return res.json();
}

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${path} failed`);
  return res.json();
}

export async function generateStory({
  character,
  sectionTitle,
  sectionTopic,
  attentionState = "Focused",
  sessionAccuracy = 0,
  eyeMetrics = null,
}) {
  const body = {
    character,
    section_title: sectionTitle,
    section_topic: sectionTopic,
    attention_state: attentionState,
    session_accuracy: sessionAccuracy,
  };
  if (eyeMetrics && typeof eyeMetrics === "object") {
    body.eye_metrics = {
      blink_rate: eyeMetrics.blink_rate,
      pupil_dilation: eyeMetrics.pupil_dilation,
      fixation_duration: eyeMetrics.fixation_duration,
      saccade_rate: eyeMetrics.saccade_rate,
      gaze_stability: eyeMetrics.gaze_stability,
    };
  }
  return post("/story/generate", body);
}

export async function generateQuestion({
  character,
  topic,
  difficulty = 2,
  attentionState = "Focused",
  questionNumber = 1,
  totalQuestions = 10,
  sessionAccuracy = 0,
  previousQuestions = [],
  eyeMetrics = null,
}) {
  const body = {
    character,
    topic,
    difficulty,
    attention_state: attentionState,
    question_number: questionNumber,
    total_questions: totalQuestions,
    session_accuracy: sessionAccuracy,
    previous_questions: previousQuestions,
  };
  if (eyeMetrics && typeof eyeMetrics === "object") {
    body.eye_metrics = {
      blink_rate: eyeMetrics.blink_rate,
      pupil_dilation: eyeMetrics.pupil_dilation,
      fixation_duration: eyeMetrics.fixation_duration,
      saccade_rate: eyeMetrics.saccade_rate,
      gaze_stability: eyeMetrics.gaze_stability,
    };
  }
  return post("/questions/generate", body);
}
// New: Language question endpoint
export async function generateLangQuestion({
  character,
  topic,
  targetLanguage = "es",
  difficulty = 2,
  attentionState = "Focused",
  questionNumber = 1,
  totalQuestions = 10,
  sessionAccuracy = 0,
  previousQuestions = [],
  eyeMetrics = null,
  skillFocus = "vocabulary_basic",
  exerciseType = null,
  learnerAge = 12,
  studentId = "default",
  sessionFatigue = 0,
}) {
  const body = {
    character,
    topic,
    target_language: targetLanguage,
    difficulty,
    attention_state: attentionState,
    question_number: questionNumber,
    total_questions: totalQuestions,
    session_accuracy: sessionAccuracy,
    previous_questions: previousQuestions,
    skill_focus: skillFocus,
    exercise_type: exerciseType,
    learner_age: learnerAge,
    student_id: studentId,
    session_fatigue: sessionFatigue,
  };
  if (eyeMetrics && typeof eyeMetrics === "object") {
    body.eye_metrics = {
      blink_rate: eyeMetrics.blink_rate,
      pupil_dilation: eyeMetrics.pupil_dilation,
      fixation_duration: eyeMetrics.fixation_duration,
      saccade_rate: eyeMetrics.saccade_rate,
      gaze_stability: eyeMetrics.gaze_stability,
    };
  }
  return post("/lang/next_question", body);
}

export async function generateFeedback({ character, totalCorrect, totalQuestions, sectionTitle, attentionHistory = [] }) {
  return post("/feedback/generate", {
    character,
    total_correct: totalCorrect,
    total_questions: totalQuestions,
    section_title: sectionTitle,
    attention_history: attentionHistory,
  });
}

export async function generateEvaluation({ character, chapterTitle, topic }) {
  return post("/evaluate/generate", {
    character,
    chapter_title: chapterTitle,
    topic,
  });
}

export async function updateProgress({ studentId = "default", sectionId, correct, responseTime = 0, attentionState = "Focused", xpEarned = 0 }) {
  return post("/progress/update", {
    student_id: studentId,
    section_id: sectionId,
    correct,
    response_time: responseTime,
    attention_state: attentionState,
    xp_earned: xpEarned,
  });
}

export async function getDashboard(studentId = "default") {
  return get(`/dashboard/${studentId}`);
}

// Language learning endpoints
export async function getLanguages() {
  return get("/lang/languages");
}

export async function generateLangStory({
  character,
  topic,
  targetLanguage = "es",
  difficulty = 2,
  learnerAge = 12,
}) {
  return post("/lang/story/generate", {
    character,
    topic,
    target_language: targetLanguage,
    difficulty,
    learner_age: learnerAge,
  });
}

export async function generateLangFeedback({
  character,
  totalCorrect,
  totalQuestions,
  targetLanguage = "es",
  topic,
  attentionHistory = [],
  skillResults = {},
  studentId = "default",
}) {
  return post("/lang/feedback/generate", {
    character,
    total_correct: totalCorrect,
    total_questions: totalQuestions,
    target_language: targetLanguage,
    topic,
    attention_history: attentionHistory,
    skill_results: skillResults,
    student_id: studentId,
  });
}

export async function updateLangProgress({
  studentId = "default",
  skillTag,
  correct,
  responseTime = 0,
  attentionState = "Focused",
  xpEarned = 0,
  exerciseType = "multiple_choice_vocab",
  targetLanguage = "es",
  sectionId = "default_section",
}) {
  return post("/lang/progress/update", {
    student_id: studentId,
    skill_tag: skillTag,
    correct,
    response_time: responseTime,
    attention_state: attentionState,
    xp_earned: xpEarned,
    exercise_type: exerciseType,
    target_language: targetLanguage,
    section_id: sectionId,
  });
}

export async function getLangDashboard(studentId = "default") {
  return get(`/lang/dashboard/${studentId}`);
}

export async function getLangMastery(studentId = "default") {
  return get(`/lang/mastery/${studentId}`);
}

export async function validateAnswer({
  learnerAnswer,
  correctAnswer,
  acceptableAnswers = [],
  targetLanguage = "es",
}) {
  return post("/lang/validate_answer", {
    learner_answer: learnerAnswer,
    correct_answer: correctAnswer,
    acceptable_answers: acceptableAnswers,
    target_language: targetLanguage,
  });
}
