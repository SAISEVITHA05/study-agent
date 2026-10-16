# quiz_engine.py
# Purpose: Quiz scoring logic — written from scratch, no AI library used
# This class tracks score, streak, and wrong answers across a quiz session

class QuizEngine:
    def __init__(self):
        """Set up a fresh quiz engine with empty state."""
        self.questions = []      # list of question dicts loaded from LLM
        self.current_index = 0   # which question the student is on
        self.score = 0           # number of correct answers
        self.streak = 0          # current consecutive correct streak
        self.best_streak = 0     # highest streak in this session
        self.wrong_log = []      # list of {"question": ..., "topic": ..., "correct_answer": ...}
        self.answer_log = []     # full history: every question + what student answered

    # ------------------------------------------------------------------
    # LOADING

    def load_questions(self, questions: list):
        """
        Load a fresh set of questions and reset all state.
        Call this every time a new quiz is generated.

        questions: list of dicts with keys:
            question, topic, options (dict A/B/C/D), answer, explanation
        """
        self.questions    = questions
        self.current_index = 0
        self.score        = 0
        self.streak       = 0
        self.best_streak  = 0
        self.wrong_log    = []
        self.answer_log   = []

    # ------------------------------------------------------------------
    # NAVIGATION
    # ------------------------------------------------------------------

    def get_current_question(self):
        """Return the current question dict, or None if quiz is over."""
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def get_question_number(self):
        """Return human-readable question number e.g. 3 (out of 5)."""
        return self.current_index + 1

    def get_total_questions(self):
        return len(self.questions)

    def is_finished(self):
        """Return True when all questions have been answered."""
        return self.current_index >= len(self.questions)

    def is_loaded(self):
        """Return True if questions have been loaded."""
        return len(self.questions) > 0

    # ------------------------------------------------------------------
    # ANSWERING
    # ------------------------------------------------------------------

    def submit_answer(self, user_answer: str):
        """
        Check the student's answer against the correct one.

        user_answer: a single letter string — "A", "B", "C", or "D"

        Returns a tuple:
            result      (str)  — "correct" or "wrong"
            correct_ans (str)  — the correct letter e.g. "B"
            explanation (str)  — why that answer is right
        """
        if self.is_finished():
            return None, None, None

        q = self.get_current_question()
        correct_ans = q["answer"].strip().upper()
        user_ans    = user_answer.strip().upper()

        # log this attempt regardless of outcome
        self.answer_log.append({
            "question":       q["question"],
            "topic":          q["topic"],
            "user_answer":    user_ans,
            "correct_answer": correct_ans,
            "was_correct":    user_ans == correct_ans
        })

        if user_ans == correct_ans:
            # correct answer
            self.score  += 1
            self.streak += 1
            if self.streak > self.best_streak:
                self.best_streak = self.streak
            result = "correct"
        else:
            # wrong answer — reset streak, log for weak topic detection
            self.streak = 0
            self.wrong_log.append({
                "question":       q["question"],
                "topic":          q["topic"],
                "correct_answer": correct_ans
            })
            result = "wrong"

        # move to next question
        self.current_index += 1

        return result, correct_ans, q["explanation"]

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------

    def get_percentage(self):
        """Return score as a percentage (0–100). Returns 0 if no questions."""
        if len(self.questions) == 0:
            return 0
        return round((self.score / len(self.questions)) * 100)

    def get_weak_topics(self):
        """
        Return a list of weak topics sorted by how many times they were wrong.
        Format: [("Cell Biology", 2), ("Thermodynamics", 1)]
        This feeds directly into the progress tracker and timetable generator.
        """
        topic_counts = {}
        for item in self.wrong_log:
            t = item["topic"]
            topic_counts[t] = topic_counts.get(t, 0) + 1
        # sort: most wrong first
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

    def get_summary(self):
        """
        Return a full summary dict of the completed quiz.
        Used by progress.py to save the session.
        """
        return {
            "score":        self.score,
            "total":        len(self.questions),
            "percent":      self.get_percentage(),
            "best_streak":  self.best_streak,
            "weak_topics":  dict(self.get_weak_topics()),
            "answer_log":   self.answer_log
        }

    def get_performance_label(self):
        """Return a friendly label based on score percentage."""
        pct = self.get_percentage()
        if pct == 100:
            return "Perfect score!"
        elif pct >= 80:
            return "Great job!"
        elif pct >= 60:
            return "Good effort, keep revising!"
        elif pct >= 40:
            return "Needs more revision"
        else:
            return "Go back to your notes and try again"
