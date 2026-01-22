import { useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const backendUrl = import.meta.env.VITE_API_URL || "";

  const ask = async () => {
    if (!question.trim()) {
      return;
    }
    setLoading(true);
    setError("");
    setAnswer("");
    try {
      const response = await fetch(`${backendUrl}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }
      const data = await response.json();
      setAnswer(data.answer || "No answer returned.");
    } catch (err) {
      setError("Could not reach the backend. Check VITE_API_URL.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div className="badge">Australia · Shark Intelligence</div>
        <h1>Shark Attacks in Australia Explained by AI</h1>
        <p>
          Ask questions about incidents, injuries, states, activities, and shark species. 
          Answers are generated from Australian Shark Attack dataset (1900 to 2026, GSAF data).
        </p>
      </header>

      <section className="panel">
        <label htmlFor="question">Your question</label>
        <textarea
          id="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="e.g., How many shark attacks were recorded in 2024?"
        />
        <div className="actions">
          <button onClick={ask} disabled={loading}>
            {loading ? "Thinking..." : "Ask the agent"}
          </button>
          <span className="hint">
            {backendUrl ? "Backend connected." : "Set VITE_API_URL to your API."}
          </span>
        </div>
      </section>

      {(answer || error) && (
        <section className="answer">
          <h2>Answer</h2>
          <p className={error ? "error" : ""}>{error || answer}</p>
        </section>
      )}
    </div>
  );
}

export default App;
