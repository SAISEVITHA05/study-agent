# prompts.py — All LLM prompts for the Study Agent project
# A: NOTE_MAKER_PROMPT, DOUBT_PROMPT
# B: QUIZ_PROMPT, FLASHCARD_PROMPT
# C: TIMETABLE_PROMPT (fallback only — main timetable uses CSP + Groq directly)

# ──B — Quiz generation ────────────────────────────────────────────────

QUIZ_PROMPT = """
Generate EXACTLY 15 MCQs.

Return ONLY JSON:
[
  {{
    "question": "...",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "answer": "A",
    "topic": "...",
    "explanation": "..."
  }}
]

Context:
{context}
"""

#  B — Flashcard generation ──────────────────────────────────────────

FLASHCARD_PROMPT = """
You are a flashcard generator. From the study material below,
extract the 8 most important concepts.

Return ONLY a valid JSON array. No extra text.

Format:
[
  {{
    "question": "Short question on one side of the card",
    "answer": "Concise answer for the other side"
  }}
]

Study material:
{context}
"""

# A — Note maker ─────────────────────────────────────────────────────

NOTE_MAKER_PROMPT = """You are a study notes generator. Your job is to generate clean, well-structured study notes.

STRICT RULES — follow all of them:
1. Use clear headings (##) and subheadings (###)
2. Use bullet points for all content
3. If a topic has a COMPARISON, LAYERS, or STEPS — embed a simple HTML table RIGHT THERE in the notes at that exact section. Do not put tables at the end.
4. For tables use exactly this format (no markdown code blocks, write raw HTML):
   <table><tr><th>Column A</th><th>Column B</th></tr><tr><td>value</td><td>value</td></tr></table>
5. Only add a table if it genuinely helps understand that topic
6. Do NOT write any explanation like "here is a table" — just put the table directly
7. Do NOT use markdown code blocks like ```html
8. Do NOT put any diagrams or tables at the very end — they must appear inline

Text to convert:
{context}"""

# ── A — Doubt answering ────────────────────────────────────────────────

DOUBT_PROMPT = """Answer this question clearly based only on the study material provided.
Question: {question}

If the answer is not in the material, say exactly: 'This topic is not covered in your uploaded material.'

Study Material:
{context}"""

# ── C — Timetable (simple fallback prompt) ─────────────────────────────

TIMETABLE_PROMPT = """You are an expert study planner.
Create a {days}-day study timetable for a student with {hours} hours per day.
Subjects: {subjects}
Weak topics that need extra time: {weak_topics}

Rules:
- Allocate more time to weak topics
- Include short breaks
- Output a clean markdown table with columns: Day | Time | Subject | Task
"""
