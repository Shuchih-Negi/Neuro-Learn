function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}
function pick(arr) {
  return arr[randInt(0, arr.length - 1)];
}

export function generateQuestion(difficulty = 1) {
  const id = crypto.randomUUID?.() ?? String(Date.now());

  if (difficulty <= 1) {
    const a = randInt(1, 20);
    const b = randInt(1, 20);
    const op = pick(["+", "-"]);
    const answer = op === "+" ? a + b : a - b;
    return { id, text: `${a} ${op} ${b} = ?`, answer, difficulty };
  }

  if (difficulty === 2) {
    const a = randInt(2, 12);
    const b = randInt(2, 12);
    const op = pick(["×", "÷"]);
    if (op === "×") return { id, text: `${a} × ${b} = ?`, answer: a * b, difficulty };
    const prod = a * b;
    return { id, text: `${prod} ÷ ${a} = ?`, answer: b, difficulty };
  }

  if (difficulty === 3) {
    const a = randInt(5, 30);
    const b = randInt(5, 30);
    const c = randInt(2, 10);
    const form = pick(["(a+b)×c", "(a-b)×c"]);
    const text = form === "(a+b)×c" ? `(${a} + ${b}) × ${c} = ?` : `(${a} - ${b}) × ${c} = ?`;
    const answer = form === "(a+b)×c" ? (a + b) * c : (a - b) * c;
    return { id, text, answer, difficulty };
  }

  if (difficulty === 4) {
    const a = randInt(2, 9);
    const x = randInt(-10, 10);
    const b = randInt(-10, 10);
    const c = a * x + b;
    const sign = b >= 0 ? `+ ${b}` : `- ${Math.abs(b)}`;
    return { id, text: `Solve: ${a}x ${sign} = ${c}`, answer: x, difficulty };
  }

  const denom = pick([2, 3, 4, 5, 6, 8, 10, 12]);
  const num = randInt(1, denom - 1);
  const k = randInt(2, 9);
  const answer = (k * num) / denom;
  return { id, text: `What is ${k} × ${num}/${denom}? (decimal)`, answer, difficulty };
}