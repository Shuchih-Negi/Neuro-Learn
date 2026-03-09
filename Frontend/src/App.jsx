import { useState } from "react";
import PixelCursor from "./components/PixelCursor";
import HomePage from "./pages/HomePage";
import RoadmapPage from "./pages/RoadmapPage";
import LearnPage from "./pages/LearnPage";
import TestPage from "./pages/TestPage";
import FeedbackPage from "./pages/FeedbackPage";
import DashboardPage from "./pages/DashboardPage";

// pages: home | roadmap | learn | test | feedback | finalboss | dashboard
export default function App() {
  const [page, setPage] = useState("home");
  const [character, setCharacter] = useState(null);
  const [chapter, setChapter] = useState(null);
  const [section, setSection] = useState(null);
  const [testResults, setTestResults] = useState(null);
  const [sectionProgress, setSectionProgress] = useState({});

  const goHome = () => { setPage("home"); setChapter(null); setSection(null); };

  const goRoadmap = () => setPage("roadmap");

  const handleSelectChapter = (ch) => {
    setChapter(ch);
    setPage("roadmap");
  };

  const handleLearn = (sec) => {
    setSection(sec);
    setPage("learn");
  };

  const handleTest = (sec) => {
    setSection(sec);
    setPage("test");
  };

  const handleTestFinish = (results) => {
    setTestResults(results);
    // Update local section progress
    if (section) {
      setSectionProgress((prev) => {
        const old = prev[section.id] || { answered: 0, correct: 0 };
        return {
          ...prev,
          [section.id]: {
            answered: old.answered + results.totalQuestions,
            correct: old.correct + results.totalCorrect,
          },
        };
      });
    }
    setPage("feedback");
  };

  const handleFinalBoss = () => {
    // Reuse test page with final boss section
    setSection({
      id: "final_boss",
      title: chapter?.finalBoss?.title || "Final Challenge",
      topic: chapter?.topic || "linear equations in two variables",
      description: "Mastery evaluation",
    });
    setPage("test");
  };

  return (
    <>
      <PixelCursor />
      {page === "home" && (
        <HomePage
          selectedCharacter={character}
          onSelectCharacter={setCharacter}
          onSelectChapter={handleSelectChapter}
          onDashboard={() => setPage("dashboard")}
        />
      )}
      {page === "roadmap" && chapter && character && (
        <RoadmapPage
          chapter={chapter}
          character={character}
          sectionProgress={sectionProgress}
          onLearn={handleLearn}
          onTest={handleTest}
          onFinalBoss={handleFinalBoss}
          onBack={goHome}
        />
      )}
      {page === "learn" && section && character && (
        <LearnPage
          section={section}
          character={character}
          onGoTest={() => handleTest(section)}
          onBack={goRoadmap}
        />
      )}
      {page === "test" && section && character && (
        <TestPage
          section={section}
          character={character}
          onFinish={handleTestFinish}
          onBack={goRoadmap}
        />
      )}
      {page === "feedback" && section && character && testResults && (
        <FeedbackPage
          section={section}
          character={character}
          results={testResults}
          onContinue={goRoadmap}
        />
      )}
      {page === "dashboard" && (
        <DashboardPage onBack={goHome} />
      )}
    </>
  );
}
