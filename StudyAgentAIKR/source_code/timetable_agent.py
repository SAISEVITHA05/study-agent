"""
timetable_agent.py — Hybrid CSP + LLM Timetable Agent (Person C)

TRUE HYBRID ARCHITECTURE — algorithm and LLM each do what they're good at:

  ALGORITHM (Python, no LLM):
    - Priority scoring via forward chaining rule engine (rule_engine.py)
    - Time allocation proportional to rule-engine priority score
    - Multiple free slot support (e.g. 9–11am AND 3–5pm)
    - Greedy CSP block placement with automatic breaks
    - Subject splitting across multiple slots

  LLM (Groq — llama-3.3-70b-versatile):
    - Writes the specific study TASK for each block
    - Adds targeted advice for weak topics
    - LLM never touches times, durations, or subject order — algorithm owns all of that

  WEEKLY MODE (updated):
    - Each day has its own independently configured time slots and subjects
    - Rule engine re-runs per day so urgency increases as exam approaches
    - Per-day slots passed as a dict: { "2026-05-02": ["09:00-11:00", ...], ... }

  RULE ENGINE INTEGRATION:
    - calculate_priority_score() is REMOVED — rule_engine.run_inference() replaces it
    - rank_subjects() is REMOVED — rule_engine.working_memory_to_ranked() replaces it
    - The working memory from the rule engine carries fired-rule reasons into blocks
    - Agent override (quiz data beats self-report) is enforced inside the rule engine
"""

import os
from datetime import date, datetime, timedelta

import groq  # pyright: ignore[reportMissingImports]

from progress import get_weak_topics, get_average_score
from rule_engine import run_inference, working_memory_to_ranked


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY SCORING — now fully delegated to the rule engine
# ─────────────────────────────────────────────────────────────────────────────

def score_and_rank_subjects(subject_entries: list) -> tuple:
    """
    Run the forward chaining rule engine over all subjects and return:
      - ranked_subjects : list of subject dicts with 'score' and 'reasons' populated
      - working_memory  : the WorkingMemory object (for UI display of fired rules)
    """
    wm = run_inference(subject_entries)
    ranked = working_memory_to_ranked(subject_entries, wm)
    return ranked, wm


# ─────────────────────────────────────────────────────────────────────────────
# TIME ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────

def allocate_time(ranked_subjects: list, total_available_mins: float) -> list:
    """
    Distribute available minutes across subjects proportionally by priority score.
    One subject can take at most 60% of total time.
    Minimum allocation: 20 minutes. Rounded to nearest 15 minutes.
    """
    total_score = sum(s["score"] for s in ranked_subjects)
    max_mins    = total_available_mins * 0.60

    if total_score == 0:
        per = total_available_mins / max(len(ranked_subjects), 1)
        for s in ranked_subjects:
            s["allocated_mins"] = max(20, round(per / 5) * 5)
        return ranked_subjects

    for s in ranked_subjects:
        raw = (s["score"] / total_score) * total_available_mins
        s["allocated_mins"] = max(20, round(min(raw, max_mins) / 15) * 15)

    return ranked_subjects


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — CSP BLOCK BUILDER
# ─────────────────────────────────────────────────────────────────────────────

STUDY_CHUNK_MINS = 50
SHORT_BREAK      = 10
LONG_BREAK       = 20


def _fmt(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def _parse_slot(slot_str: str) -> tuple:
    start_str, end_str = slot_str.strip().split("-")
    start = datetime.strptime(start_str.strip(), "%H:%M")
    end   = datetime.strptime(end_str.strip(),   "%H:%M")
    return start, end


def build_time_blocks(ranked_subjects: list, free_slots: list) -> list:
    """
    PHASE 1 — Greedy CSP scheduler with multiple free slot support.
    free_slots: list of strings like ['09:00-11:00', '14:00-16:00']
    """
    parsed_slots = []
    for s in free_slots:
        try:
            start, end = _parse_slot(s)
            if end > start:
                parsed_slots.append((start, end))
        except Exception:
            continue

    if not parsed_slots:
        return []

    total_avail_mins = sum(
        (e - s).seconds // 60 for s, e in parsed_slots
    )

    ranked_with_time = allocate_time(ranked_subjects, total_avail_mins)

    chunks = []
    for s in ranked_with_time:
        remaining = s["allocated_mins"]
        while remaining > 0:
            chunk_size = min(remaining, STUDY_CHUNK_MINS)
            chunks.append({
                "subject":       s["subject"],
                "duration_mins": chunk_size,
                "task_type":     s.get("task_type", "general_study"),
                "reasons":       s.get("reasons", ""),
            })
            remaining -= chunk_size

    blocks            = []
    consecutive_study = 0
    chunk_idx         = 0

    for (slot_start, slot_end) in parsed_slots:
        cursor = slot_start

        blocks.append({
            "start":         _fmt(slot_start),
            "end":           _fmt(slot_end),
            "type":          "slot_header",
            "subject":       f"Free window: {_fmt(slot_start)} – {_fmt(slot_end)}",
            "duration_mins": 0,
            "task_type":     "slot_header",
            "reasons":       "",
        })

        while chunk_idx < len(chunks):
            chunk = chunks[chunk_idx]
            dur   = chunk["duration_mins"]

            if cursor + timedelta(minutes=dur) > slot_end:
                mins_left = (slot_end - cursor).seconds // 60
                mins_left = (mins_left // 15) * 15
                if mins_left >= 20:
                    partial_end = cursor + timedelta(minutes=mins_left)
                    blocks.append({
                        "start":         _fmt(cursor),
                        "end":           _fmt(partial_end),
                        "type":          "study",
                        "subject":       chunk["subject"],
                        "duration_mins": mins_left,
                        "task_type":     chunk["task_type"],
                        "reasons":       chunk["reasons"],
                    })
                    cursor = partial_end
                    consecutive_study += mins_left
                    chunks[chunk_idx]["duration_mins"] -= mins_left
                    if chunks[chunk_idx]["duration_mins"] <= 0:
                        chunk_idx += 1
                break

            if consecutive_study >= STUDY_CHUNK_MINS:
                break_dur = LONG_BREAK if consecutive_study >= STUDY_CHUNK_MINS * 2 else SHORT_BREAK
                break_end = cursor + timedelta(minutes=break_dur)
                if break_end > slot_end:
                    break
                blocks.append({
                    "start":         _fmt(cursor),
                    "end":           _fmt(break_end),
                    "type":          "break",
                    "subject":       "Break",
                    "duration_mins": break_dur,
                    "task_type":     "break",
                    "reasons":       "",
                })
                cursor = break_end
                consecutive_study = 0

            block_end = cursor + timedelta(minutes=dur)
            blocks.append({
                "start":         _fmt(cursor),
                "end":           _fmt(block_end),
                "type":          "study",
                "subject":       chunk["subject"],
                "duration_mins": dur,
                "task_type":     chunk["task_type"],
                "reasons":       chunk["reasons"],
            })
            cursor = block_end
            consecutive_study += dur
            chunk_idx += 1

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — GROQ PERSONALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(blocks: list, weak_topics: list) -> str:
    subject_names = [b["subject"] for b in blocks if b["type"] == "study"]
    weak_names = [w[0] for w in weak_topics if w[0] in subject_names]

    block_lines = ""
    for b in blocks:
        if b["type"] == "slot_header":
            block_lines += f"\n  [{b['subject']}]\n"
        elif b["type"] == "break":
            block_lines += f"  - {b['start']}–{b['end']} | BREAK ({b['duration_mins']} min)\n"
        else:
            flag = " ← WEAK TOPIC" if b["subject"] in weak_names else ""
            block_lines += (
                f"  - {b['start']}–{b['end']} | {b['subject']} "
                f"({b['duration_mins']} min, {b['task_type'].replace('_', ' ')}){flag}\n"
            )

    weak_block = (
        f"Subjects flagged as weak by quiz performance: {', '.join(weak_names)}"
        if weak_names else "No weak subjects flagged yet."
    )

    return f"""You are an expert study coach. The TIME SLOTS below were generated by a scheduling algorithm — do NOT change any times, durations, or subject order.

Your only job: write one specific, actionable study task for each study block (one sentence each).

{weak_block}

TIMETABLE (algorithm-generated):
{block_lines}

Output a Markdown table with exactly these four columns:
| Time | Subject | Duration | What to do |

Rules:
- Copy the Time, Subject, and Duration columns exactly as shown — do not modify them
- For BREAK rows: "What to do" = "Rest, stretch, or drink water"
- For slot_header rows: skip them entirely — do not include them in the table
- For weak topics: suggest targeted practice (e.g. "Attempt 5 past-paper questions on Limits")
- For exam prep: suggest practice over passive reading
- For assignments: focus on the specific deliverable
- Output ONLY the markdown table — no intro text, no notes after
"""


def personalise_with_llm(blocks: list, api_key: str) -> str:
    weak_topics = get_weak_topics()
    prompt      = _build_prompt(blocks, weak_topics)

    try:
        client = groq.Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise study coach. Follow instructions exactly. Output only what is asked."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=1500,
        )

        return response.choices[0].message.content

    except Exception as e:
        lines = [
            "| Time | Subject | Duration | What to do |",
            "|------|---------|----------|------------|"
        ]
        for b in blocks:
            if b["type"] == "break":
                lines.append(
                    f"| {b['start']}–{b['end']} | Break | {b['duration_mins']} min | Rest, stretch, or drink water |"
                )
            elif b["type"] == "study":
                lines.append(
                    f"| {b['start']}–{b['end']} | {b['subject']} | {b['duration_mins']} min | Study {b['subject']} |"
                )
        return (
            "\n".join(lines) +
            f"\n\n> ⚠️ Groq unavailable: {e}. Showing algorithm output only."
        )


# ─────────────────────────────────────────────────────────────────────────────
# WEEKLY MODE — updated to support per-day slots and subjects
# ─────────────────────────────────────────────────────────────────────────────

def build_weekly_plan(
    daily_config: dict,
    exam_date: date,
    api_key: str,
) -> tuple:
    """
    Generate a week's worth of timetables, one section per day.

    daily_config: dict keyed by date string, each value is:
      {
        "subject_entries": [...],   # subjects for that specific day
        "free_slots": [...]         # time slots for that specific day
      }

    The rule engine re-runs per day so urgency correctly increases
    as the exam date approaches.

    Returns (full_markdown, all_blocks_by_date, last_working_memory)
    """
    today     = date.today()
    days_left = max((exam_date - today).days, 1)
    output    = ""
    all_blocks = {}
    last_wm    = None

    for i in range(days_left):
        current_day = today + timedelta(days=i)
        date_str    = str(current_day)
        day_label   = current_day.strftime("%A, %d %B")

        # Get this day's config — fall back to empty if not configured
        day_cfg          = daily_config.get(date_str, {})
        day_subjects     = day_cfg.get("subject_entries", [])
        day_slots        = day_cfg.get("free_slots", [])

        # Skip days with no configuration
        if not day_subjects or not day_slots:
            output += f"\n## {day_label}\n\n_No subjects or slots configured for this day._\n"
            continue

        # Rule engine runs fresh per day — urgency grows as exam approaches
        ranked_subjects, wm = score_and_rank_subjects(day_subjects)
        last_wm = wm

        blocks = build_time_blocks(ranked_subjects, day_slots)
        all_blocks[date_str] = blocks

        output += f"\n## {day_label}\n\n"
        output += personalise_with_llm(blocks, api_key)
        output += "\n"

    return output, all_blocks, last_wm


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_timetable(
    subject_entries: list,
    free_slots: list,
    mode: str,
    api_key: str,
    exam_date: date = None,
    daily_config: dict = None,
) -> tuple:
    """
    Full pipeline:
      1. Run forward chaining rule engine → ranked subjects + working memory
      2. Feed ranked subjects into CSP block builder → time blocks
      3. Feed blocks into Groq → personalised markdown timetable

    For weekly mode, pass daily_config (dict of date → {subject_entries, free_slots}).
    If daily_config is not provided in weekly mode, falls back to using the same
    subject_entries and free_slots for every day (old behaviour).

    Returns (markdown_string, blocks, working_memory).
    """
    if mode == "weekly" and exam_date:
        if daily_config:
            # New per-day configuration
            markdown, all_blocks, wm = build_weekly_plan(daily_config, exam_date, api_key)
            # Flatten blocks for parse_blocks_to_tasks compatibility
            flat_blocks = []
            for day_blocks in all_blocks.values():
                flat_blocks.extend(day_blocks)
            return markdown, flat_blocks, wm
        else:
            # Fallback: same config every day (old behaviour)
            ranked_subjects, wm = score_and_rank_subjects(subject_entries)
            blocks = []
            today = date.today()
            days_left = max((exam_date - today).days, 1)
            output = ""
            for i in range(days_left):
                current_day = today + timedelta(days=i)
                day_label   = current_day.strftime("%A, %d %B")
                day_blocks  = build_time_blocks(ranked_subjects, free_slots)
                output += f"\n## {day_label}\n\n"
                output += personalise_with_llm(day_blocks, api_key)
                output += "\n"
                blocks.extend(day_blocks)
            return output, blocks, wm
    else:
        # Daily mode — unchanged
        ranked_subjects, wm = score_and_rank_subjects(subject_entries)
        blocks   = build_time_blocks(ranked_subjects, free_slots)
        markdown = personalise_with_llm(blocks, api_key)
        return markdown, blocks, wm


def parse_blocks_to_tasks(blocks: list) -> list:
    """Convert CSP block list → task format for progress.py."""
    return [
        {
            "start_time":       b["start"],
            "subject":          b["subject"],
            "task_type":        b["task_type"],
            "duration_mins":    b["duration_mins"],
            "completed":        False,
            "rescheduled_from": None,
        }
        for b in blocks
        if b["type"] in ("study", "break")
    ]