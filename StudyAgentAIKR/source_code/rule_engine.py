"""
rule_engine.py — Forward Chaining Rule Engine (Person C)

Directly implements Unit 5: Reasoning Fundamentals — Forward Chaining.

HOW IT WORKS:
  1. Facts are loaded from progress.json (Unit 4: Knowledge Base)
  2. Rules are defined as condition → action pairs (Production Rules)
  3. The engine fires rules in order, adding new facts to the working memory
  4. Final working memory contains derived conclusions used by the timetable agent

This is a simplified Rete-style forward chaining engine.
Each rule checks a condition over the fact base and asserts new facts if true.
"""

from datetime import date
from progress import get_weak_topics, get_average_score, get_upcoming_deadlines


# ─────────────────────────────────────────────────────────────────────────────
# WORKING MEMORY — the agent's current belief state
# Starts empty, gets populated by facts + rule firings
# ─────────────────────────────────────────────────────────────────────────────

class WorkingMemory:
    def __init__(self):
        self.facts = {}       # fact_name → value
        self.fired_rules = [] # log of which rules fired (for UI display)

    def assert_fact(self, key: str, value):
        self.facts[key] = value

    def get(self, key, default=None):
        return self.facts.get(key, default)

    def has(self, key) -> bool:
        return key in self.facts

    def log_rule(self, rule_name: str, reason: str):
        self.fired_rules.append({"rule": rule_name, "reason": reason})


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION RULES — each rule is: IF condition THEN assert new fact
# These map directly to Unit 5: Reasoning with Horn Clauses
# Each rule is a Horn clause: body → head
# ─────────────────────────────────────────────────────────────────────────────

def rule_critical_exam(wm: WorkingMemory, subject: str):
    """
    RULE: IF subject has exam AND days_until_exam <= 2 AND score < 60
          THEN assert CRITICAL priority
    """
    avg   = get_average_score(subject)
    deads = get_upcoming_deadlines(days_ahead=14)

    for d in deads:
        if d["subject"].lower() == subject.lower() and d["type"] == "Exam":
            if d["days_away"] <= 2 and (avg is None or avg < 60):
                wm.assert_fact(f"{subject}_priority", "CRITICAL")
                wm.assert_fact(f"{subject}_time_multiplier", 2.0)
                wm.log_rule(
                    "critical_exam",
                    f"{subject}: exam in {d['days_away']}d, avg {avg or 'unknown'}%"
                )
                return True
    return False


def rule_weak_topic(wm: WorkingMemory, subject: str):
    """
    RULE: IF quiz_average(subject) < 60%
          THEN assert WEAK flag AND increase time allocation
    """
    avg = get_average_score(subject)
    if avg is not None and avg < 60:
        wm.assert_fact(f"{subject}_is_weak", True)
        wm.assert_fact(f"{subject}_time_multiplier",
                       max(wm.get(f"{subject}_time_multiplier", 1.0), 1.5))
        wm.log_rule("weak_topic", f"{subject}: avg score {avg}% < 60%")
        return True
    return False


def rule_user_override(wm: WorkingMemory, subject: str, intellect: str):
    """
    RULE: IF user says intellect=weak AND subject not already CRITICAL
          THEN assert WEAK flag (user self-report)
    Agent override: if quiz data contradicts user, quiz data wins.
    """
    avg = get_average_score(subject)

    # Quiz data overrides self-report — this is the agent intelligence
    if avg is not None and avg >= 80 and intellect == "weak":
        wm.assert_fact(f"{subject}_self_report_overridden", True)
        wm.log_rule(
            "agent_override",
            f"{subject}: user says weak but quiz avg={avg}% — override applied"
        )
        return False  # rule did NOT fire — agent rejected user's claim

    if intellect == "weak" and not wm.has(f"{subject}_is_weak"):
        wm.assert_fact(f"{subject}_is_weak", True)
        wm.assert_fact(f"{subject}_time_multiplier",
                       max(wm.get(f"{subject}_time_multiplier", 1.0), 1.3))
        wm.log_rule("user_override", f"{subject}: self-reported as weak")
        return True
    return False


def rule_deadline_urgency(wm: WorkingMemory, subject: str, deadline: date):
    """
    RULE: IF deadline exists AND days_until <= 3
          THEN assert URGENT and increase time multiplier
    """
    if deadline is None:
        return False

    days_left = (deadline - date.today()).days
    if days_left <= 1:
        wm.assert_fact(f"{subject}_urgent", True)
        wm.assert_fact(f"{subject}_time_multiplier",
                       max(wm.get(f"{subject}_time_multiplier", 1.0), 2.5))
        wm.log_rule("deadline_urgency", f"{subject}: deadline TOMORROW")
        return True
    elif days_left <= 3:
        wm.assert_fact(f"{subject}_urgent", True)
        wm.assert_fact(f"{subject}_time_multiplier",
                       max(wm.get(f"{subject}_time_multiplier", 1.0), 1.8))
        wm.log_rule("deadline_urgency", f"{subject}: deadline in {days_left} days")
        return True
    return False


def rule_well_prepared(wm: WorkingMemory, subject: str, preparedness: int):
    """
    RULE: IF preparedness >= 8 AND no CRITICAL flag
          THEN reduce time allocation (student doesn't need much time here)
    """
    if preparedness is None:
        return False
    if preparedness >= 8 and not wm.get(f"{subject}_priority") == "CRITICAL":
        current = wm.get(f"{subject}_time_multiplier", 1.0)
        wm.assert_fact(f"{subject}_time_multiplier", max(0.5, current * 0.7))
        wm.log_rule("well_prepared", f"{subject}: preparedness {preparedness}/10 — reducing time")
        return True
    return False


def rule_cascade_critical(wm: WorkingMemory, subject: str):
    """
    RULE: IF subject is CRITICAL AND subject is WEAK
          THEN assert EMERGENCY — this subject must dominate the timetable
    This is a chained rule — fires only if two other rules already fired.
    Demonstrates rule chaining (key feature of forward chaining engines).
    """
    if (wm.get(f"{subject}_priority") == "CRITICAL" and
            wm.get(f"{subject}_is_weak")):
        wm.assert_fact(f"{subject}_time_multiplier", 3.0)
        wm.assert_fact(f"{subject}_status", "EMERGENCY")
        wm.log_rule(
            "cascade_critical",
            f"{subject}: EMERGENCY — critical exam + weak topic combined"
        )
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD CHAINING ENGINE — runs all rules in order for each subject
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(subjects: list) -> WorkingMemory:
    wm = WorkingMemory()

    # Phase 1 — assert base facts from knowledge base (progress.json)
    weak_topics = get_weak_topics()
    for w in weak_topics:
        # get_weak_topics() returns (topic_name, wrong_count) tuples
        wm.assert_fact(f"{w[0]}_quiz_avg", w[1])

    # Phase 2 — fire rules for each subject
    for s in subjects:
        name         = s["subject"]
        intellect    = s.get("intellect")
        preparedness = s.get("preparedness")
        deadline     = s.get("deadline_obj")

        rule_critical_exam(wm, name)
        rule_weak_topic(wm, name)
        rule_user_override(wm, name, intellect)
        rule_deadline_urgency(wm, name, deadline)
        rule_well_prepared(wm, name, preparedness)
        rule_cascade_critical(wm, name)

        multiplier = wm.get(f"{name}_time_multiplier", 1.0)
        base_score = _base_score(s)
        final_score = round(base_score * multiplier)
        wm.assert_fact(f"{name}_final_score", final_score)

    return wm


def _base_score(subject_entry: dict) -> float:
    """Simple base score before rule multipliers are applied."""
    score = 10  # everyone starts at 10
    task_type = subject_entry.get("task_type", "general_study")
    if task_type == "exam":
        score += 20
    elif task_type == "assignment":
        score += 10
    return score


def working_memory_to_ranked(subjects: list, wm: WorkingMemory) -> list:
    """
    Convert working memory facts back into the ranked subject list
    that the CSP scheduler expects.
    """
    result = []
    for s in subjects:
        name = s["subject"]
        entry = dict(s)
        entry["score"]   = wm.get(f"{name}_final_score", 10)
        entry["reasons"] = _build_reason_string(name, wm)
        result.append(entry)

    return sorted(result, key=lambda x: x["score"], reverse=True)


def _build_reason_string(subject: str, wm: WorkingMemory) -> str:
    """Build a human-readable reason string from working memory facts."""
    parts = []
    if wm.get(f"{subject}_status") == "EMERGENCY":
        parts.append("EMERGENCY: critical exam + weak topic")
    elif wm.get(f"{subject}_priority") == "CRITICAL":
        parts.append("CRITICAL exam deadline")
    if wm.get(f"{subject}_is_weak"):
        avg = wm.get(f"{subject}_quiz_avg")
        parts.append(f"weak topic (avg {avg}%)" if avg else "weak topic")
    if wm.get(f"{subject}_urgent"):
        parts.append("urgent deadline")
    if wm.get(f"{subject}_self_report_overridden"):
        parts.append("agent overrode self-report")
    return ", ".join(parts) if parts else "general study"