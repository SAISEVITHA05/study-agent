# progress.py — Unified Progress Tracker
# Merges quiz session tracking with timetable tracking
# Written from scratch — no database library, plain JSON only
#
# save_session(subject, summary)
# save_timetable(), get_timetable(), mark_task_complete(), etc.
# Dashboard calls: get_weak_topics(), get_scores_by_subject(), get_streak(), etc.

import json
import os
from datetime import date, datetime, timedelta

# Single shared file for the whole project
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "progress.json")

WEAK_THRESHOLD   = 60
STRONG_THRESHOLD = 80


# INTERNAL HELPERS


def _load() -> dict:
    if not os.path.exists(DATA_FILE):
        return {
            "sessions":      [],
            "weak_topics":   {},
            "timetables":    {},
            "subjects_meta": {}
        }
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def _save(data: dict):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


# B INTERFACE — called after every quiz submission to save results and update weak topics

def save_session(subject: str, summary: dict):
    """
    Called by B's quiz engine after every quiz submission.
    summary = quiz_engine.get_summary()
    """
    data = _load()

    data["sessions"].append({
        "id":          f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "date":        str(date.today()),
        "subject":     subject,
        "score":       summary["score"],
        "total":       summary["total"],
        "percent":     summary["percent"],
        "best_streak": summary.get("best_streak", 0),
        "session_type": "quiz"
    })

    # Update cumulative weak topics (B's format)
    for topic, count in summary.get("weak_topics", {}).items():
        data["weak_topics"][topic] = data["weak_topics"].get(topic, 0) + count

    _save(data)



# B's PROGRESS READ FUNCTIONS — for dashboard analytics and weak topic tracking

def load_progress():
    return _load()


def get_all_sessions():
    return _load()["sessions"]


def get_weak_topics(top_n=5):
    """
    Used by: progress page AND Person C's timetable agent.
    """
    data = _load()
    sorted_topics = sorted(
        data["weak_topics"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_topics[:top_n]


def get_scores_by_subject():
    """Returns {subject: avg_percent} dict for bar chart."""
    sessions = get_all_sessions()
    subject_totals = {}
    for s in sessions:
        subj = s["subject"]
        if subj not in subject_totals:
            subject_totals[subj] = []
        subject_totals[subj].append(s["percent"])
    return {subj: round(sum(scores) / len(scores))
            for subj, scores in subject_totals.items()}


def get_total_sessions():
    return len(get_all_sessions())


def get_overall_average():
    sessions = get_all_sessions()
    if not sessions:
        return 0
    return round(sum(s["percent"] for s in sessions) / len(sessions))


#C's FUNCTIONS — subject history, average scores, timetable tracking

def get_average_score(subject: str):
    """Average quiz % for a subject. Used by rule_engine and timetable_agent."""
    sessions = [s for s in get_all_sessions()
                if s["subject"].lower() == subject.lower()]
    if not sessions:
        return None
    return round(sum(s["percent"] for s in sessions) / len(sessions), 1)


def get_subject_history(subject: str) -> list:
    return sorted(
        [s for s in get_all_sessions() if s["subject"].lower() == subject.lower()],
        key=lambda x: x["date"]
    )


def get_all_subjects() -> list:
    return list(set(s["subject"] for s in get_all_sessions()))


def save_timetable(target_date: str, tasks: list, mode: str = "daily",
                   available_hours: float = 8.0):
    data = _load()
    if "timetables" not in data:
        data["timetables"] = {}
    data["timetables"][target_date] = {
        "mode":            mode,
        "available_hours": available_hours,
        "generated_at":    str(datetime.now()),
        "tasks": [
            {
                "id":              f"task_{i}_{target_date.replace('-','')}",
                "subject":         t.get("subject", ""),
                "duration_mins":   t.get("duration_mins", 60),
                "start_time":      t.get("start_time", ""),
                "task_type":       t.get("task_type", "general_study"),
                "completed":       False,
                "rescheduled_from": t.get("rescheduled_from", None)
            }
            for i, t in enumerate(tasks)
        ]
    }
    _save(data)


def get_timetable(target_date: str) -> dict:
    return _load().get("timetables", {}).get(target_date, {})


def mark_task_complete(target_date: str, task_id: str):
    data = _load()
    for task in data.get("timetables", {}).get(target_date, {}).get("tasks", []):
        if task["id"] == task_id:
            task["completed"] = True
            break
    _save(data)


def get_missed_tasks(target_date: str) -> list:
    day = get_timetable(target_date)
    return [t for t in day.get("tasks", []) if not t.get("completed", False)]


def suggest_reschedule_slots(task: dict, days_ahead: int = 7) -> list:
    today     = date.today()
    task_mins = task.get("duration_mins", 60)
    task_subj = task.get("subject", "")
    candidates = []

    for i in range(1, days_ahead + 1):
        candidate_date = today + timedelta(days=i)
        date_str       = str(candidate_date)
        day_data       = get_timetable(date_str)

        scheduled_mins   = sum(t.get("duration_mins", 0) for t in day_data.get("tasks", []))
        available_hours  = day_data.get("available_hours", 8.0)
        total_avail_mins = available_hours * 60
        free_mins        = total_avail_mins - scheduled_mins

        if free_mins < task_mins:
            continue

        same_subject_count = sum(
            1 for t in day_data.get("tasks", [])
            if t.get("subject", "").lower() == task_subj.lower()
        )

        urgency_weight  = 1 / i
        capacity_weight = free_mins / max(total_avail_mins, 1)
        overload_penalty = min(same_subject_count * 0.2, 1.0)
        slot_score = (urgency_weight * 0.5) + (capacity_weight * 0.4) - (overload_penalty * 0.1)

        candidates.append({
            "date":               date_str,
            "day_name":           candidate_date.strftime("%A, %d %B"),
            "score":              round(slot_score, 4),
            "free_mins":          int(free_mins),
            "already_has_subject": same_subject_count > 0
        })

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]


def reschedule_task(original_date: str, task_id: str, new_date: str):
    data = _load()
    task_to_move = None

    for task in data.get("timetables", {}).get(original_date, {}).get("tasks", []):
        if task["id"] == task_id:
            task["completed"]     = True
            task["rescheduled_to"] = new_date
            task_to_move = dict(task)
            break

    if not task_to_move:
        return False

    if "timetables" not in data:
        data["timetables"] = {}
    if new_date not in data["timetables"]:
        data["timetables"][new_date] = {
            "mode": "daily", "available_hours": 8.0,
            "generated_at": str(datetime.now()), "tasks": []
        }

    new_task = dict(task_to_move)
    new_task["id"]               = f"task_rescheduled_{datetime.now().strftime('%H%M%S')}"
    new_task["completed"]        = False
    new_task["rescheduled_from"] = original_date
    new_task["start_time"]       = ""
    data["timetables"][new_date]["tasks"].append(new_task)
    _save(data)
    return True


def get_streak() -> int:
    data       = _load()
    timetables = data.get("timetables", {})
    streak     = 0
    check_date = date.today() - timedelta(days=1)

    while True:
        date_str = str(check_date)
        if date_str not in timetables:
            break
        tasks = timetables[date_str].get("tasks", [])
        if not tasks:
            break
        if all(t.get("completed", False) for t in tasks):
            streak    += 1
            check_date -= timedelta(days=1)
        else:
            break
    return streak


def get_completion_rate(days: int = 7) -> list:
    data   = _load()
    result = []
    for i in range(days - 1, -1, -1):
        d        = date.today() - timedelta(days=i)
        date_str = str(d)
        day_data = data.get("timetables", {}).get(date_str, {})
        tasks    = day_data.get("tasks", [])
        total    = len(tasks)
        completed = sum(1 for t in tasks if t.get("completed", False))
        result.append({
            "date":      d.strftime("%d %b"),
            "completed": completed,
            "missed":    total - completed,
            "total":     total,
            "rate":      round((completed / total) * 100) if total > 0 else 0
        })
    return result


def get_upcoming_deadlines(days_ahead: int = 14) -> list:
    data      = _load()
    deadlines = []
    today     = date.today()

    for date_str, day_data in data.get("timetables", {}).items():
        try:
            task_date = date.fromisoformat(date_str)
        except ValueError:
            continue
        days_away = (task_date - today).days
        if 0 <= days_away <= days_ahead:
            for task in day_data.get("tasks", []):
                if task.get("task_type") in ("exam", "assignment"):
                    urgency = "red" if days_away <= 2 else ("amber" if days_away <= 7 else "green")
                    deadlines.append({
                        "subject":   task["subject"],
                        "type":      task["task_type"].title(),
                        "date":      task_date.strftime("%d %b"),
                        "days_away": days_away,
                        "urgency":   urgency
                    })

    return sorted(deadlines, key=lambda x: x["days_away"])[:5]


def get_agent_recommendation() -> str:
    weak      = get_weak_topics(top_n=3)
    deadlines = get_upcoming_deadlines(days_ahead=5)
    streak    = get_streak()

    if deadlines:
        d    = deadlines[0]
        weak_names = [w[0].lower() for w in weak]
        if d["subject"].lower() in weak_names:
            return (f"⚠️ You have a {d['type']} in {d['subject']} in {d['days_away']} day(s) "
                    f"and quiz data flags it as weak. Today's plan will prioritise this heavily.")
        return (f"📅 {d['subject']} {d['type']} is in {d['days_away']} day(s). "
                f"Your timetable has been adjusted to reflect this deadline.")

    if weak:
        names = ", ".join(w[0] for w in weak[:2])
        return (f"📉 Quiz data shows you're struggling with {names}. "
                f"Your next timetable will allocate extra time here.")

    if streak >= 3:
        return f"🔥 {streak}-day streak! Keep it up — consistency is the key to exam success."

    return "✅ All caught up! Upload notes or run a quiz to let the agent help you plan smarter."
