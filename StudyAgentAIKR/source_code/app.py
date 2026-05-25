"""
app.py — Integrated Study Agent

Run with: streamlit run source_code/app.py
"""

import os
import json
import sys
import streamlit as st
from dotenv import load_dotenv

# ── Path setup so all imports work ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Load API key from .env automatically ─────────────────────────────────────
load_dotenv(override=True)
if os.getenv("GROQ_API_KEY"):
    st.session_state["api_key"] = os.getenv("GROQ_API_KEY")

# ── Imports ───────────────────────────────────────────────────────────────────
from groq import Groq
from progress import (
    save_session, get_weak_topics, get_scores_by_subject,
    get_total_sessions, get_overall_average,
    get_all_subjects, get_subject_history, get_average_score,
    get_streak, get_completion_rate, get_upcoming_deadlines,
    get_agent_recommendation, get_timetable, save_timetable,
    mark_task_complete, get_missed_tasks,
    suggest_reschedule_slots, reschedule_task, _load
)
from quiz_engine import QuizEngine
from prompts import QUIZ_PROMPT, FLASHCARD_PROMPT, NOTE_MAKER_PROMPT, DOUBT_PROMPT, TIMETABLE_PROMPT
from timetable_agent import (
    score_and_rank_subjects,
    build_time_blocks,
    generate_timetable, parse_blocks_to_tasks
)

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import date, datetime, timedelta
import fitz  # PyMuPDF for Person A's PDF reading
import re

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Global CSS from style.css
css_path = os.path.join(os.path.dirname(__file__), "..", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Nunito:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Nunito', sans-serif !important; }
    </style>
    """, unsafe_allow_html=True)

# Ambient Stars + Sparkles
st.markdown('''
  <div class="orb-field">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
  </div>
  <div class="stars-canvas">
    <div class="star star-1"></div><div class="star star-2"></div>
    <div class="star star-3"></div><div class="star star-4"></div>
    <div class="star star-5"></div><div class="star star-6"></div>
    <div class="star star-7"></div><div class="star star-8"></div>
    <div class="star star-9"></div><div class="star star-10"></div>
    <div class="star star-11"></div><div class="star star-12"></div>
  </div>
  <div class="sparkle-field">
    <div class="sparkle sparkle-1"></div><div class="sparkle sparkle-2"></div>
    <div class="sparkle sparkle-3"></div><div class="sparkle sparkle-4"></div>
    <div class="sparkle sparkle-5"></div><div class="sparkle sparkle-6"></div>
    <div class="sparkle sparkle-7"></div><div class="sparkle sparkle-8"></div>
  </div>
''', unsafe_allow_html=True)

# API key and Groq client
api_key = st.session_state.get("api_key", "")
groq_client = Groq(api_key=api_key) if api_key else None

def call_llm(prompt: str) -> str:
    if not groq_client:
        return "⚠️ API key not set. Add GROQ_API_KEY to your .env file."
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful study assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# RAG / PDF helpers

def split_chunks(text, size=800):
    sentences = text.replace('\n', ' ').split('. ')
    chunks, current, count = [], [], 0
    for s in sentences:
        current.append(s)
        count += len(s)
        if count >= size:
            chunks.append('. '.join(current))
            current, count = [], 0
    if current:
        chunks.append('. '.join(current))
    return chunks

def find_relevant(question, chunks, top_k=3):
    keywords = set(re.sub(r'[^\w\s]', '', question.lower()).split())
    scored   = sorted([(len(keywords & set(c.lower().split())), c) for c in chunks], reverse=True)
    return "\n\n".join([c for _, c in scored[:top_k]])

# Session state defaults
defaults = {
    "engine":          QuizEngine(),
    "query_engine":    None,
    "quiz_subject":    "",
    "last_result":     None,
    "notes_output":    "",
    "flashcards":      [],
    "fc_index":        0,
    "fc_flipped":      False,
    "pdf_text":        None,
    "pdf_chunks":      None,
    "pdf_name":        None,
    "answer":          None,
    "history":         [],
    "subject_entries": [],
    "free_slots":      [],
    "last_timetable":  None,
    "last_blocks":     [],
    "weekly_day_config": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar
with st.sidebar:
    st.markdown('<div class="page-title" style="font-size:1.6rem;">🧠 Study Agent</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📄 Upload your notes (PDF)", type=["pdf", "txt"])
    if uploaded_file:
        if (st.session_state.pdf_text is None or
                st.session_state.pdf_name != uploaded_file.name):
            with st.spinner("Reading PDF..."):
                if uploaded_file.name.endswith(".pdf"):
                    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    st.session_state.pdf_text = "".join(p.get_text() for p in pdf)
                else:
                    st.session_state.pdf_text = uploaded_file.read().decode("utf-8")
                st.session_state.pdf_chunks = split_chunks(st.session_state.pdf_text)
                st.session_state.pdf_name   = uploaded_file.name
                st.session_state.notes_output = ""
                st.session_state.answer       = None
        st.success(f"✅ {uploaded_file.name} ready")

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📝 Notes", "💬 Doubt", "🧠 Quiz",
         "🃏 Flashcards", "📊 Progress", "📈 Dashboard", "📅 Timetable"]
    )
    st.markdown("---")
    st.caption("Agent Status: **Online** 🟢")

def require_pdf():
    if st.session_state.pdf_text is None:
        st.warning("Upload your notes PDF using the sidebar first.")
        st.stop()


# ── HOME ──────────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    st.markdown('<div class="page-title">🧠 Study Agent</div>', unsafe_allow_html=True)
    st.markdown("##### An intelligent study companion that learns from you — and plans for you.")
    st.markdown("---")

    st.markdown("""
    <p style="color:var(--text-muted);font-size:1.05rem;line-height:1.9;max-width:820px;">
    Study Agent is a fully integrated AI study system. It reads your notes, tests your understanding,
    tracks where you struggle, and then builds a personalised timetable — automatically prioritising
    the subjects you need most, using a <strong style="color:var(--accent-violet)">forward chaining rule engine</strong>
    and a <strong style="color:var(--accent-cyan)">constraint satisfaction scheduler</strong>.
    Everything is connected: what you score in the quiz directly shapes what the timetable gives you more time for.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('''
        <div class="glass-card">
          <h3>📄 RAG Notes &amp; Doubt</h3>
          <h4 style="color:var(--accent-cyan);margin-top:0">Your PDF, Made Useful</h4>
          <p style="color:var(--text-muted);font-size:0.92rem;line-height:1.65;">
            Upload any study PDF. The agent extracts structured notes with headings and tables.
            Ask any question about your material — it retrieves the most relevant chunks and answers from them.
            This is RAG: Retrieval-Augmented Generation.
          </p>
        </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown('''
        <div class="glass-card">
          <h3>🧠 Quiz &amp; Flashcards</h3>
          <h4 style="color:var(--accent-violet);margin-top:0">Test, Track, Improve</h4>
          <p style="color:var(--text-muted);font-size:0.92rem;line-height:1.65;">
            Generates 15 MCQs directly from your notes. Tracks your score, streak, and
            which topics you keep getting wrong. Weak topics are saved automatically to the
            knowledge base — the timetable agent reads them to prioritise your schedule.
          </p>
        </div>
        ''', unsafe_allow_html=True)
    with c3:
        st.markdown('''
        <div class="glass-card">
          <h3>📅 Smart Timetable</h3>
          <h4 style="color:var(--accent-pink);margin-top:0">Priority-Driven Planning</h4>
          <p style="color:var(--text-muted);font-size:0.92rem;line-height:1.65;">
            A forward chaining rule engine scores every subject using your quiz history,
            deadlines, self-reported difficulty, and preparedness level. A CSP scheduler
            then allocates your free time proportionally — weakest, most urgent subjects
            get the most slots. Groq (Llama 3.3) personalises the task for each block.
          </p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c4, c5 = st.columns(2)
    with c4:
        st.markdown('''
        <div class="glass-card">
          <h3>📈 Progress Dashboard</h3>
          <h4 style="color:var(--accent-cyan);margin-top:0">See the Whole Picture</h4>
          <p style="color:var(--text-muted);font-size:0.92rem;line-height:1.65;">
            Visualises your quiz scores over time, task completion rate across the last 7 days,
            study time by subject, day streak, upcoming deadlines, and agent-detected weak topics —
            all in one live dashboard that updates as you study.
          </p>
        </div>
        ''', unsafe_allow_html=True)
    with c5:
        st.markdown('''
        <div class="glass-card">
          <h3>🔁 Missed Task Rescheduler</h3>
          <h4 style="color:var(--accent-pink);margin-top:0">Never Fall Behind</h4>
          <p style="color:var(--text-muted);font-size:0.92rem;line-height:1.65;">
            Missed a study block? The agent finds the best free slot in your next 7 days
            and lets you reschedule with one click. Today's task list tracks completion
            live — check off tasks and the agent marks your progress automatically.
          </p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### How everything connects")
    st.markdown("""
- Upload your notes → quiz questions are generated directly from your material
- Quiz scores + wrong answers → saved to the knowledge base automatically
- The rule engine reads your weak topics, deadlines and preparedness → scores and ranks every subject
- The CSP scheduler turns those scores into a time-allocated study plan across your free slots
- The dashboard reflects all of it — scores, completion, streaks, weak topics — in real time
    """)


# ── NOTES ─────────────────────────────────────────────────────────────────────

elif page == "📝 Notes":
    st.markdown('<div class="page-title">📝 Note Maker</div>', unsafe_allow_html=True)
    require_pdf()

    if st.button("✨ Generate My Notes!", type="primary"):
        with st.spinner("Generating notes from your PDF..."):
            prompt = NOTE_MAKER_PROMPT.format(context=st.session_state.pdf_text[:15000])
            st.session_state.notes_output = call_llm(prompt)

    if st.session_state.notes_output:
        st.markdown(
            f'<div class="notes-box">{st.session_state.notes_output}</div>',
            unsafe_allow_html=True
        )
        st.download_button(
            "⬇️ Download notes as .txt",
            st.session_state.notes_output,
            file_name="my_notes.txt"
        )


# ── DOUBT ─────────────────────────────────────────────────────────────────────

elif page == "💬 Doubt":
    st.markdown('<div class="page-title">💬 Ask a Doubt</div>', unsafe_allow_html=True)
    require_pdf()

    st.markdown("🔍 **Ask anything** from your uploaded PDF!")
    question = st.text_input("💬 Your question:", placeholder="e.g. What is the OSI model?")

    if st.button("💡 Get Answer!", type="primary"):
        if question.strip():
            with st.spinner("Searching your PDF..."):
                relevant = find_relevant(question, st.session_state.pdf_chunks)
                prompt   = DOUBT_PROMPT.format(context=relevant, question=question)
                st.session_state.answer = call_llm(prompt)
                st.session_state.history.append({
                    "type": "doubt", "label": question,
                    "content": st.session_state.answer
                })
        else:
            st.warning("Type a question first.")

    if st.session_state.answer:
        st.markdown(
            f'<div class="answer-box">{st.session_state.answer}</div>',
            unsafe_allow_html=True
        )


# ── QUIZ ──────────────────────────────────────────────────────────────────────

elif page == "🧠 Quiz":
    st.markdown('<div class="page-title">🧠 Quiz Maker</div>', unsafe_allow_html=True)
    require_pdf()
    engine = st.session_state.engine

    col1, col2 = st.columns([3, 1])
    with col1:
        subject = st.text_input("Subject / topic", placeholder="e.g. Cell Biology, Chapter 3")
    with col2:
        st.write("")
        generate = st.button("Generate Quiz", type="primary", use_container_width=True)

    if generate:
        if not subject.strip():
            st.error("Enter a subject name first.")
        else:
            with st.spinner("Generating questions from your notes..."):
                relevant = find_relevant(subject, st.session_state.pdf_chunks)
                prompt   = QUIZ_PROMPT.format(context=relevant)
                raw      = call_llm(prompt).strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
                try:
                    questions = json.loads(raw)
                    engine.load_questions(questions)
                    st.session_state.quiz_subject = subject
                    st.session_state.last_result  = None
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Couldn't parse questions — try again.")

    st.divider()

    if not engine.is_loaded():
        st.info("Enter a subject above and click Generate Quiz to begin.")
        st.stop()

    if engine.is_finished():
        summary = engine.get_summary()

        # ── CONFETTI BLAST ────────────────────────────────────────────
        import streamlit.components.v1 as components
        components.html("""
<script>
(function() {
  function launch() {
    if (document.getElementById('aikr-cc')) return;
    const COLORS = ['#7c6dfa','#f472b6','#22d3ee','#fbbf24','#34d399',
                    '#fb923c','#c084fc','#f87171','#818cf8','#a3e635'];
    const cv = document.createElement('canvas');
    cv.id = 'aikr-cc';
    cv.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:2147483647;';
    document.body.appendChild(cv);
    const ctx = cv.getContext('2d');
    cv.width = window.innerWidth;
    cv.height = window.innerHeight;
    let P = [], END = Date.now() + 2000;
    function rand(a,b){return Math.random()*(b-a)+a;}
    function make(){
      const s=Math.random();
      let x,vx;
      if(s<0.33){x=rand(0,cv.width*0.25);vx=rand(3,8);}
      else if(s<0.66){x=rand(cv.width*0.75,cv.width);vx=rand(-8,-3);}
      else{x=rand(0,cv.width);vx=rand(-4,4);}
      return{x,y:-10,vx,vy:rand(5,12),sz:rand(8,16),
             c:COLORS[Math.floor(Math.random()*COLORS.length)],
             r:rand(0,6.28),rs:rand(-0.2,0.2),
             sh:Math.floor(Math.random()*3),
             w:rand(0,6.28),ws:rand(0.06,0.15),op:1};
    }
    function star(r){
      ctx.beginPath();
      for(let i=0;i<10;i++){
        const a=(i*Math.PI/5)-Math.PI/2,rd=i%2?r*0.42:r;
        i===0?ctx.moveTo(rd*Math.cos(a),rd*Math.sin(a)):ctx.lineTo(rd*Math.cos(a),rd*Math.sin(a));
      }
      ctx.closePath();ctx.fill();
    }
    function loop(){
      const rem=END-Date.now();
      ctx.clearRect(0,0,cv.width,cv.height);
      if(rem>0&&P.length<300){
        const n=rem>1500?20:6;
        for(let i=0;i<n;i++)P.push(make());
      }
      P.forEach(p=>{
        p.w+=p.ws;p.x+=p.vx+Math.sin(p.w)*2;
        p.y+=p.vy;p.vy+=0.12;p.r+=p.rs;
        if(rem<=0)p.op=Math.max(0,p.op-0.025);
        ctx.save();ctx.globalAlpha=p.op;
        ctx.translate(p.x,p.y);ctx.rotate(p.r);
        ctx.fillStyle=p.c;
        if(p.sh===0){ctx.fillRect(-p.sz/2,-p.sz/3,p.sz,p.sz/2.5);}
        else if(p.sh===1){ctx.beginPath();ctx.arc(0,0,p.sz/2.5,0,6.28);ctx.fill();}
        else{star(p.sz/2);}
        ctx.restore();
      });
      P=P.filter(p=>p.y<cv.height+60&&p.op>0);
      if(rem>0||P.length)requestAnimationFrame(loop);
      else cv.remove();
    }
    loop();
    window.addEventListener('resize',()=>{cv.width=window.innerWidth;cv.height=window.innerHeight;});
  }

  // Try launching in parent window first, fall back to current window
  try {
    const w = window.parent;
    if (w && w !== window) {
      const s = w.document.createElement('script');
      s.textContent = '(' + launch.toString() + ')()';
      w.document.body.appendChild(s);
      s.remove();
    } else { launch(); }
  } catch(e) { launch(); }
})();
</script>
""", height=0, scrolling=False)
        # ── END CONFETTI ──────────────────────────────────────────────
        st.subheader("Quiz Complete! 🎉")
        m1, m2, m3 = st.columns(3)
        m1.metric("Score",       f"{summary['score']} / {summary['total']}")
        m2.metric("Percentage",  f"{summary['percent']}%")
        m3.metric("Best Streak", f"{summary['best_streak']} in a row")
        st.info(engine.get_performance_label())
        weak_q = engine.get_weak_topics()
        if weak_q:
            st.warning("**Topics to revise more:**")
            for topic, count in weak_q:
                st.markdown(f'<span class="weak-tag">⚠ {topic} — {count} wrong</span>',
                            unsafe_allow_html=True)
        else:
            st.success("No weak topics — perfect!")
        save_session(st.session_state.quiz_subject, summary)
        st.caption("✅ Results saved to progress tracker. Dashboard and Timetable will update.")
        with st.expander("Review all answers"):
            for i, log in enumerate(summary["answer_log"]):
                icon = "✅" if log["was_correct"] else "❌"
                st.write(f"{icon} **Q{i+1}:** {log['question']}")
                if not log["was_correct"]:
                    st.write(f"   You answered **{log['user_answer']}** — correct was **{log['correct_answer']}**")
        if st.button("Take Another Quiz"):
            engine.load_questions([])
            st.rerun()
        st.stop()

    q   = engine.get_current_question()
    num = engine.get_question_number()
    tot = engine.get_total_questions()
    st.progress(num / tot, text=f"Question {num} of {tot}  |  Score: {engine.score}/{num-1 if num>1 else 0}")

    if st.session_state.last_result:
        result, correct, explanation = st.session_state.last_result
        if result == "correct":
            st.success(f"✅ Correct!  Streak: {engine.streak} 🔥")
        else:
            st.error(f"❌ Wrong. Correct answer was **{correct}**")
        st.info(f"💡 {explanation}")
        st.session_state.last_result = None

    st.markdown(f'''
    <div class="quiz-card">
      <div class="quiz-question-text">Q{num}: {q["question"]}</div>
      <div style="font-size:0.78rem;color:var(--accent-violet);font-weight:600;
                  letter-spacing:0.06em;text-transform:uppercase;">
        Topic: {q["topic"]}
      </div>
    </div>
    ''', unsafe_allow_html=True)

    choice = st.radio(
        "Choose your answer:",
        options=list(q["options"].keys()),
        format_func=lambda k: f"{k}:  {q['options'][k]}",
        key=f"radio_q{num}"
    )
    if st.button("Submit Answer", type="primary"):
        result, correct, explanation = engine.submit_answer(choice)
        st.session_state.last_result = (result, correct, explanation)
        st.rerun()


# ── FLASHCARDS ────────────────────────────────────────────────────────────────

elif page == "🃏 Flashcards":
    st.markdown('<div class="page-title">🃏 Flashcards</div>', unsafe_allow_html=True)
    require_pdf()

    if st.button("Generate Flashcards", type="primary"):
        with st.spinner("Creating flashcards..."):
            relevant = find_relevant("key concepts and definitions", st.session_state.pdf_chunks)
            prompt   = FLASHCARD_PROMPT.format(context=relevant)
            raw      = call_llm(prompt).strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            try:
                st.session_state.flashcards = json.loads(raw)
                st.session_state.fc_index   = 0
                st.session_state.fc_flipped = False
                st.rerun()
            except json.JSONDecodeError:
                st.error("Couldn't parse flashcards. Try again.")

    cards = st.session_state.flashcards
    if cards:
        idx     = st.session_state.fc_index
        card    = cards[idx]
        flipped = st.session_state.fc_flipped
        st.caption(f"Card {idx + 1} of {len(cards)}")

        flip_class = "flashcard-scene flipped" if flipped else "flashcard-scene"
        st.markdown(f'''
        <div class="{flip_class}" onclick="">
          <div class="flashcard-inner">
            <div class="flashcard-front">
              <div>
                <div style="font-size:0.72rem;color:var(--accent-violet);
                            font-weight:700;letter-spacing:0.1em;
                            text-transform:uppercase;margin-bottom:12px;">Question</div>
                {card["question"]}
              </div>
            </div>
            <div class="flashcard-back">
              <div>
                <div style="font-size:0.72rem;color:var(--accent-green);
                            font-weight:700;letter-spacing:0.1em;
                            text-transform:uppercase;margin-bottom:12px;">Answer</div>
                {card["answer"]}
              </div>
            </div>
          </div>
        </div>
        ''', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⬅ Prev") and idx > 0:
                st.session_state.fc_index  -= 1
                st.session_state.fc_flipped = False
                st.rerun()
        with c2:
            if st.button("🔄 Flip Card"):
                st.session_state.fc_flipped = not flipped
                st.rerun()
        with c3:
            if st.button("Next ➡") and idx < len(cards) - 1:
                st.session_state.fc_index  += 1
                st.session_state.fc_flipped = False
                st.rerun()


# ── PROGRESS ──────────────────────────────────────────────────────────────────

elif page == "📊 Progress":
    st.markdown('<div class="page-title">📊 Your Progress</div>', unsafe_allow_html=True)
    total_sessions = get_total_sessions()
    if total_sessions == 0:
        st.info("No quiz sessions yet. Take a quiz first!")
        st.stop()
    overall_avg       = get_overall_average()
    weak_topics_list  = get_weak_topics(top_n=5)
    scores_by_subject = get_scores_by_subject()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sessions",   total_sessions)
    m2.metric("Overall Average",  f"{overall_avg}%")
    m3.metric("Subjects Studied", len(scores_by_subject))
    st.divider()
    if scores_by_subject:
        st.subheader("Average score by subject")
        st.bar_chart(scores_by_subject)
    if weak_topics_list:
        st.subheader("Topics needing revision")
        for topic, count in weak_topics_list:
            colour = "🔴" if count >= 3 else "🟡" if count == 2 else "🟠"
            st.markdown(
                f'<span class="weak-tag">{colour} {topic} — got wrong {count} time(s)</span>',
                unsafe_allow_html=True
            )
    else:
        st.success("No weak topics yet — great work!")


# ── DASHBOARD ─────────────────────────────────────────────────────────────────

elif page == "📈 Dashboard":
    st.markdown('<div class="page-title">📈 Dashboard</div>', unsafe_allow_html=True)
    st.markdown("##### What the agent knows about your progress")
    st.markdown("---")

    all_subjects    = get_all_subjects()
    weak            = get_weak_topics(top_n=10)
    streak          = get_streak()
    completion_data = get_completion_rate(days=7)
    avg_completion  = sum(d["rate"] for d in completion_data) / max(len(completion_data), 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Subjects tracked",   len(all_subjects))
    col2.metric("Weak topics",        len(weak))
    col3.metric("Day streak 🔥",       streak)
    col4.metric("Avg completion (7d)", f"{avg_completion:.0f}%")
    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-header">Quiz scores over time</div>', unsafe_allow_html=True)
        if not all_subjects:
            st.info("No quiz data yet. Take a quiz first — scores will appear here.")
        else:
            rows = []
            for subject in all_subjects:
                for session in get_subject_history(subject):
                    rows.append({"Date": session["date"], "Subject": subject,
                                 "Score (%)": session["percent"]})
            if rows:
                df  = pd.DataFrame(rows)
                fig = px.line(df, x="Date", y="Score (%)", color="Subject",
                              markers=True,
                              color_discrete_sequence=["#7c6dfa","#f472b6","#22d3ee","#34d399","#fbbf24"])
                fig.add_hline(y=60, line_dash="dash",
                              line_color="rgba(255,100,100,0.5)",
                              annotation_text="Weak threshold (60%)",
                              annotation_position="bottom right")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  margin=dict(l=0,r=0,t=10,b=0),
                                  font=dict(family="Nunito"), height=300)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Task completion — last 7 days</div>',
                    unsafe_allow_html=True)
        cdf  = pd.DataFrame(completion_data)
        fig2 = go.Figure()
        fig2.add_bar(name="Completed", x=cdf["date"], y=cdf["completed"],
                     marker_color="rgba(52,211,153,0.7)")
        fig2.add_bar(name="Missed",    x=cdf["date"], y=cdf["missed"],
                     marker_color="rgba(248,113,113,0.6)")
        fig2.update_layout(barmode="stack", plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0,r=0,t=10,b=0),
                           font=dict(family="Nunito"), height=220)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Agent insight</div>', unsafe_allow_html=True)
        rec = get_agent_recommendation()
        st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Weak topics (agent-detected)</div>',
                    unsafe_allow_html=True)
        if not weak:
            st.markdown('<span class="strong-tag">✓ No weak topics</span>',
                        unsafe_allow_html=True)
        else:
            for w in weak:
                st.markdown(
                    f'<span class="weak-tag">⚠ {w[0]} — {w[1]} wrong</span>',
                    unsafe_allow_html=True
                )

        st.markdown('<div class="section-header">Upcoming deadlines</div>',
                    unsafe_allow_html=True)
        deadlines = get_upcoming_deadlines()
        if not deadlines:
            st.caption("No upcoming deadlines.")
        else:
            for d in deadlines:
                urgency_class = f"urgency-{d['urgency']}"
                st.markdown(
                    f'<span class="{urgency_class}">📅 {d["subject"]} {d["type"]} — '
                    f'{d["date"]} ({d["days_away"]}d away)</span>',
                    unsafe_allow_html=True
                )

        data         = _load()
        subject_mins = {}
        for day_data in data.get("timetables", {}).values():
            for task in day_data.get("tasks", []):
                s = task.get("subject", "Unknown")
                if task.get("task_type") not in ("break", "slot_header"):
                    subject_mins[s] = subject_mins.get(s, 0) + task.get("duration_mins", 0)
        if subject_mins:
            st.markdown('<div class="section-header">Study time by subject</div>',
                        unsafe_allow_html=True)
            donut_df = pd.DataFrame([{"Subject": k, "Minutes": v}
                                     for k, v in subject_mins.items()])
            fig3 = px.pie(donut_df, names="Subject", values="Minutes", hole=0.55,
                          color_discrete_sequence=["#7c6dfa","#f472b6","#22d3ee","#34d399","#fbbf24","#fb923c"])
            fig3.update_layout(margin=dict(l=0,r=0,t=10,b=0),
                               paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Nunito"), height=240)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="section-header">Current streak</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="streak-badge">{streak}</span>&nbsp; consecutive days',
                    unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()


# ── TIMETABLE ─────────────────────────────────────────────────────────────────

elif page == "📅 Timetable":
    st.markdown('<div class="page-title">📅 Timetable</div>', unsafe_allow_html=True)
    st.markdown("##### CSP scheduler + Forward chaining rule engine + Groq personalisation")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["✨ Generate Plan", "✅ Today's Tasks", "🔁 Missed Tasks"])

    # Generate Plan
    with tab1:
        col_form, col_result = st.columns([2, 3])

        with col_form:
            st.markdown('<div class="section-header">Plan mode</div>', unsafe_allow_html=True)
            mode     = st.radio("Mode", ["📅 Daily", "📆 Weekly"], horizontal=True)
            mode_key = "daily" if "Daily" in mode else "weekly"

            exam_date = None
            if mode_key == "weekly":
                exam_date = st.date_input(
                    "Exam / target date",
                    value=date.today() + timedelta(days=7),
                    min_value=date.today() + timedelta(days=1)
                )

            if mode_key == "daily":
                st.markdown('<div class="section-header">Your free time slots</div>',
                            unsafe_allow_html=True)
                sc1, sc2 = st.columns(2)
                with sc1:
                    slot_start = st.time_input("Start", value=datetime.strptime("09:00","%H:%M").time())
                with sc2:
                    slot_end = st.time_input("End",   value=datetime.strptime("11:00","%H:%M").time())

                if st.button("➕ Add slot"):
                    s_str = slot_start.strftime("%H:%M")
                    e_str = slot_end.strftime("%H:%M")
                    if slot_end > slot_start:
                        label = f"{s_str}-{e_str}"
                        if label not in st.session_state.free_slots:
                            st.session_state.free_slots.append(label)
                    else:
                        st.warning("End must be after start.")

                for i, slot in enumerate(st.session_state.free_slots):
                    c1, c2 = st.columns([4, 1])
                    c1.markdown(f'<span class="slot-pill">🕐 {slot}</span>',
                                unsafe_allow_html=True)
                    if c2.button("✕", key=f"ds_{i}"):
                        st.session_state.free_slots.pop(i)
                        st.rerun()

                st.markdown('<div class="section-header">Add subjects</div>', unsafe_allow_html=True)

                with st.expander("➕ Add a subject", expanded=True):
                    s_name    = st.text_input("Subject", placeholder="e.g. Physics")
                    task_type = st.selectbox("Type", ["General Study","Assignment","Exam"])
                    intellect    = None
                    preparedness = None
                    deadline_obj = None

                    if task_type == "General Study":
                        intellect = st.select_slider(
                            "Your level", options=["Weak","Average","Strong"],
                            value="Average"
                        ).lower()
                    else:
                        preparedness = st.slider("Preparedness (1–10)", 1, 10, 5)
                        deadline_obj = st.date_input(
                            "Deadline",
                            value=date.today() + timedelta(days=3),
                            min_value=date.today()
                        )

                    if st.button("Add Subject"):
                        if s_name.strip():
                            entry = {
                                "subject":      s_name.strip(),
                                "task_type":    task_type.lower().replace(" ", "_"),
                                "intellect":    intellect,
                                "preparedness": preparedness,
                                "deadline_obj": deadline_obj,
                                "deadline":     str(deadline_obj) if deadline_obj else None,
                                "score":        0,
                                "reasons":      "pending rule engine",
                            }
                            st.session_state.subject_entries.append(entry)
                            st.success(f"Added {s_name.strip()}")
                        else:
                            st.warning("Enter a subject name.")

                if st.session_state.subject_entries:
                    st.markdown('<div class="section-header">Queued subjects</div>',
                                unsafe_allow_html=True)
                    for i, e in enumerate(st.session_state.subject_entries):
                        c1, c2 = st.columns([5, 1])
                        c1.markdown(
                            f'<span class="subject-chip">{e["subject"]}</span>'
                            f'<small style="opacity:0.5">{e.get("reasons","")}</small>',
                            unsafe_allow_html=True
                        )
                        if c2.button("✕", key=f"de_{i}"):
                            st.session_state.subject_entries.pop(i)
                            st.rerun()

            else:
                st.markdown('<div class="section-header">Configure each day</div>',
                            unsafe_allow_html=True)
                st.caption("Expand each day to set its own time slots and subjects independently.")

                today     = date.today()
                days_left = max((exam_date - today).days, 1)

                for i in range(days_left):
                    current_day = today + timedelta(days=i)
                    date_str    = str(current_day)
                    day_label   = current_day.strftime("%A, %d %B")

                    if date_str not in st.session_state.weekly_day_config:
                        st.session_state.weekly_day_config[date_str] = {
                            "subject_entries": [],
                            "free_slots":      []
                        }

                    day_cfg = st.session_state.weekly_day_config[date_str]

                    with st.expander(f"📆 {day_label}", expanded=(i == 0)):
                        st.markdown("**⏰ Time slots**")
                        wsc1, wsc2 = st.columns(2)
                        with wsc1:
                            w_start = st.time_input(
                                "Start", key=f"wstart_{date_str}",
                                value=datetime.strptime("09:00", "%H:%M").time()
                            )
                        with wsc2:
                            w_end = st.time_input(
                                "End", key=f"wend_{date_str}",
                                value=datetime.strptime("11:00", "%H:%M").time()
                            )

                        if st.button("➕ Add slot", key=f"wadd_slot_{date_str}"):
                            s_str = w_start.strftime("%H:%M")
                            e_str = w_end.strftime("%H:%M")
                            if w_end > w_start:
                                label = f"{s_str}-{e_str}"
                                if label not in day_cfg["free_slots"]:
                                    day_cfg["free_slots"].append(label)
                                    st.rerun()
                            else:
                                st.warning("End must be after start.")

                        for si, slot in enumerate(day_cfg["free_slots"]):
                            wc1, wc2 = st.columns([4, 1])
                            wc1.markdown(f'<span class="slot-pill">🕐 {slot}</span>',
                                         unsafe_allow_html=True)
                            if wc2.button("✕", key=f"wdel_slot_{date_str}_{si}"):
                                day_cfg["free_slots"].pop(si)
                                st.rerun()

                        st.markdown("---")
                        st.markdown("**📚 Subjects for this day**")

                        ws_name    = st.text_input("Subject", key=f"wsub_{date_str}",
                                                    placeholder="e.g. Maths")
                        w_task_type = st.selectbox("Type", ["General Study","Assignment","Exam"],
                                                    key=f"wtype_{date_str}")
                        w_intellect    = None
                        w_preparedness = None
                        w_deadline_obj = None

                        if w_task_type == "General Study":
                            w_intellect = st.select_slider(
                                "Your level", options=["Weak","Average","Strong"],
                                value="Average", key=f"wlevel_{date_str}"
                            ).lower()
                        else:
                            w_preparedness = st.slider(
                                "Preparedness (1–10)", 1, 10, 5,
                                key=f"wprep_{date_str}"
                            )
                            w_deadline_obj = st.date_input(
                                "Deadline", key=f"wdeadline_{date_str}",
                                value=date.today() + timedelta(days=3),
                                min_value=date.today()
                            )

                        if st.button("Add Subject", key=f"wadd_sub_{date_str}"):
                            if ws_name.strip():
                                entry = {
                                    "subject":      ws_name.strip(),
                                    "task_type":    w_task_type.lower().replace(" ", "_"),
                                    "intellect":    w_intellect,
                                    "preparedness": w_preparedness,
                                    "deadline_obj": w_deadline_obj,
                                    "deadline":     str(w_deadline_obj) if w_deadline_obj else None,
                                    "score":        0,
                                    "reasons":      "pending rule engine",
                                }
                                day_cfg["subject_entries"].append(entry)
                                st.success(f"Added {ws_name.strip()} to {day_label}")
                                st.rerun()
                            else:
                                st.warning("Enter a subject name.")

                        if day_cfg["subject_entries"]:
                            for ei, e in enumerate(day_cfg["subject_entries"]):
                                ec1, ec2 = st.columns([5, 1])
                                ec1.markdown(
                                    f'<span class="subject-chip">{e["subject"]}</span>'
                                    f'<small style="opacity:0.5"> {e["task_type"].replace("_"," ")}</small>',
                                    unsafe_allow_html=True
                                )
                                if ec2.button("✕", key=f"wdel_sub_{date_str}_{ei}"):
                                    day_cfg["subject_entries"].pop(ei)
                                    st.rerun()
                        else:
                            st.caption("No subjects added for this day yet.")

            st.markdown("---")

            if mode_key == "daily":
                can_gen = (bool(st.session_state.subject_entries) and
                           bool(st.session_state.free_slots) and
                           bool(api_key))
            else:
                any_day_ready = any(
                    cfg.get("subject_entries") and cfg.get("free_slots")
                    for cfg in st.session_state.weekly_day_config.values()
                )
                can_gen = any_day_ready and bool(api_key)

            if not api_key:
                st.warning("GROQ_API_KEY not found in .env file.")
            if mode_key == "daily" and not st.session_state.free_slots:
                st.info("Add at least one free slot.")

            if st.button("🧠 Generate Timetable", disabled=not can_gen,
                         type="primary", use_container_width=True):
                with st.spinner("Rule engine + CSP running... then Groq personalises..."):

                    if mode_key == "daily":
                        markdown, blocks, wm = generate_timetable(
                            subject_entries=st.session_state.subject_entries,
                            free_slots=st.session_state.free_slots,
                            mode=mode_key,
                            api_key=api_key,
                            exam_date=None,
                        )
                        tasks = parse_blocks_to_tasks(blocks)
                        save_timetable(str(date.today()), tasks, mode_key)

                    else:
                        markdown, blocks, wm = generate_timetable(
                            subject_entries=[],
                            free_slots=[],
                            mode=mode_key,
                            api_key=api_key,
                            exam_date=exam_date,
                            daily_config=st.session_state.weekly_day_config,
                        )
                        from timetable_agent import build_time_blocks, score_and_rank_subjects
                        for date_str, day_cfg in st.session_state.weekly_day_config.items():
                            if day_cfg.get("subject_entries") and day_cfg.get("free_slots"):
                                ranked, _ = score_and_rank_subjects(day_cfg["subject_entries"])
                                day_blocks = build_time_blocks(ranked, day_cfg["free_slots"])
                                day_tasks  = parse_blocks_to_tasks(day_blocks)
                                existing = get_timetable(date_str)
                                if not existing.get("tasks"):
                                    save_timetable(date_str, day_tasks, "weekly")

                    st.session_state.last_timetable = markdown
                    st.session_state.last_blocks    = blocks

                    if wm and wm.fired_rules:
                        st.session_state["fired_rules"] = wm.fired_rules

        with col_result:
            st.markdown('<div class="section-header">Generated plan</div>',
                        unsafe_allow_html=True)
            if st.session_state.last_timetable:
                st.caption("⚡ Rule engine → CSP algorithm → Groq (Llama 3.3)")
                st.markdown(st.session_state.last_timetable)

                if st.session_state.get("fired_rules"):
                    with st.expander("🔍 Rule firing trace (forward chaining log)"):
                        for r in st.session_state["fired_rules"]:
                            st.markdown(
                                f'<div class="rule-trace"><strong><code>{r["rule"]}</code></strong>'
                                f' — {r["reason"]}</div>',
                                unsafe_allow_html=True
                            )
            else:
                st.markdown("""
                <div style="opacity:0.35;text-align:center;margin-top:80px;line-height:2.2;">
                    <div style="font-size:2.8rem;">📋</div>
                    <div>Your timetable will appear here</div>
                    <div style="font-size:0.85rem;">Add slots + subjects, then Generate</div>
                </div>""", unsafe_allow_html=True)

    # Today's Tasks
    with tab2:
        today_str  = str(date.today())
        today_data = get_timetable(today_str)
        tasks      = today_data.get("tasks", [])
        st.markdown(f"**{date.today().strftime('%A, %d %B %Y')}**")

        if not tasks:
            st.info("No timetable for today. Generate one above.")
        else:
            done = sum(1 for t in tasks if t.get("completed", False))
            st.progress(done / len(tasks), text=f"{done}/{len(tasks)} tasks done")
            for task in tasks:
                if task.get("task_type") in ("break","slot_header"):
                    st.caption(f"☕ {task.get('start_time','')} Break")
                    continue
                c1, c2, c3 = st.columns([1, 6, 2])
                task_done  = task.get("completed", False)
                with c1:
                    if st.checkbox("", value=task_done,
                                   key=f"chk_{task['id']}", disabled=task_done):
                        mark_task_complete(today_str, task["id"])
                        st.rerun()
                with c2:
                    t_str = f"**{task['start_time']}** " if task.get("start_time") else ""
                    st.markdown(
                        f"{t_str}"
                        f'<span class="subject-chip">{task["subject"]}</span>'
                        f"{'~~' if task_done else ''}"
                        f"{task.get('task_type','').replace('_',' ').title()}"
                        f"{'~~' if task_done else ''}",
                        unsafe_allow_html=True
                    )
                with c3:
                    st.caption(f"{task.get('duration_mins','?')} min {'✓' if task_done else ''}")

            real_tasks = [t for t in tasks if t.get("task_type") not in ("break","slot_header")]
            if done == len(real_tasks) and len(real_tasks) > 0:
                st.success("🎉 All done for today!")

    # Missed Tasks
    with tab3:
        check_date = st.date_input(
            "Check missed tasks for",
            value=date.today() - timedelta(days=1),
            max_value=date.today()
        )
        missed = get_missed_tasks(str(check_date))
        if not missed:
            st.success(f"✓ No missed tasks on {check_date.strftime('%d %B')}.")
        else:
            st.warning(f"{len(missed)} task(s) missed on {check_date.strftime('%d %B %Y')}")
            for task in missed:
                if task.get("task_type") in ("break","slot_header"):
                    continue
                with st.expander(f"📌 {task['subject']} — {task.get('duration_mins','?')} min"):
                    slots = suggest_reschedule_slots(task)
                    if not slots:
                        st.warning("No free slots found in next 7 days.")
                    else:
                        st.markdown("**Top 3 suggested slots:**")
                        for rank, slot in enumerate(slots, 1):
                            ci, cb = st.columns([4, 1])
                            ci.markdown(
                                f"**#{rank} {slot['day_name']}** — "
                                f"{slot['free_mins']} min free · score {slot['score']}"
                            )
                            if cb.button("Move", key=f"rs_{task['id']}_{slot['date']}"):
                                if reschedule_task(str(check_date), task["id"], slot["date"]):
                                    st.success(f"Moved to {slot['day_name']}!")
                                    st.rerun()