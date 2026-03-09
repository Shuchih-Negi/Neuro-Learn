import batmanImg from "../assets/batman.png";
import jokerImg from "../assets/joker.png";
import alfredImg from "../assets/alfred.png";

export const characters = [
  { id: "batman", name: "Batman", image: batmanImg, personality: "strategic and calm" },
  { id: "joker", name: "Joker", image: jokerImg, personality: "playful and unpredictable" },
  { id: "alfred", name: "Alfred", image: alfredImg, personality: "wise and encouraging" },
];

export const chapters = [
  {
    id: "linear_eq_2var",
    title: "Linear Equations with Two Variables",
    description: "Master the art of solving equations with two unknowns through epic quests",
    topic: "linear equations in two variables",
    sections: [
      {
        id: "intro",
        title: "Introduction to Linear Equations",
        topic: "basic concepts of linear equations in two variables, what ax+by=c means, identifying linear equations",
        description: "Learn what linear equations look like and why they matter",
      },
      {
        id: "graphing",
        title: "Graphing on Coordinate Plane",
        topic: "plotting linear equations on the coordinate plane, finding x and y intercepts, slope of a line",
        description: "Plot equations and see them come alive on graphs",
      },
      {
        id: "pair_of_equations",
        title: "Pair of Linear Equations",
        topic: "understanding systems of two linear equations, consistent vs inconsistent systems, dependent systems",
        description: "What happens when two lines meet? Discover systems of equations",
      },
      {
        id: "substitution",
        title: "Solving by Substitution",
        topic: "substitution method for solving a pair of linear equations in two variables step by step",
        description: "Replace and conquer — solve systems by substituting one variable",
      },
      {
        id: "elimination",
        title: "Solving by Elimination",
        topic: "elimination method for solving a pair of linear equations in two variables step by step",
        description: "Add or subtract equations to eliminate a variable and find the answer",
      },
      {
        id: "word_problems",
        title: "Real-World Word Problems",
        topic: "word problems that translate to a pair of linear equations in two variables, real-life applications",
        description: "Apply your skills to solve real-world problems with two unknowns",
      },
    ],
    finalBoss: {
      title: "Final Boss — The Ultimate Challenge",
      description: "5 mastery questions to prove you've conquered this chapter!",
      questionCount: 5,
    },
  },
];
