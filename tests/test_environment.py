"""
Tests for Email Triage Environment
====================================
Run: pytest tests/ -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from email_triage_env.server.environment import (
    EmailTriageEnvironment,
    EmailAction,
    EmailObservation,
    _grade_episode,
    _score_reply_quality,
)
from email_triage_env.server.email_data import TASKS, EMAIL_CORPUS


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    return EmailTriageEnvironment(task_name="easy_triage")

@pytest.fixture
def medium_env():
    return EmailTriageEnvironment(task_name="medium_triage")

@pytest.fixture
def hard_env():
    return EmailTriageEnvironment(task_name="hard_triage")


# ── Task configuration tests ────────────────────────────────────────────────────

def test_all_tasks_defined():
    assert "easy_triage" in TASKS
    assert "medium_triage" in TASKS
    assert "hard_triage" in TASKS

def test_task_difficulties():
    assert TASKS["easy_triage"]["difficulty"] == "easy"
    assert TASKS["medium_triage"]["difficulty"] == "medium"
    assert TASKS["hard_triage"]["difficulty"] == "hard"

def test_task_email_counts():
    assert len(TASKS["easy_triage"]["email_ids"]) == 5
    assert len(TASKS["medium_triage"]["email_ids"]) == 8
    assert len(TASKS["hard_triage"]["email_ids"]) == 12

def test_all_email_ids_exist():
    corpus_ids = {e["id"] for e in EMAIL_CORPUS}
    for task_name, task in TASKS.items():
        for eid in task["email_ids"]:
            assert eid in corpus_ids, f"Email {eid} in task {task_name} not in corpus"

def test_corpus_has_required_fields():
    required = {"id", "from", "to", "subject", "body", "true_priority", "true_category",
                "true_action", "requires_response"}
    for email in EMAIL_CORPUS:
        for field in required:
            assert field in email, f"Email {email['id']} missing field '{field}'"

def test_corpus_valid_priorities():
    valid = {"urgent", "high", "medium", "low"}
    for email in EMAIL_CORPUS:
        assert email["true_priority"] in valid

def test_corpus_valid_actions():
    valid = {"reply", "archive", "delete", "escalate"}
    for email in EMAIL_CORPUS:
        assert email["true_action"] in valid


# ── reset() tests ───────────────────────────────────────────────────────────────

def test_reset_returns_observation(easy_env):
    obs = easy_env.reset(seed=42)
    assert isinstance(obs, EmailObservation)
    assert obs.done is False
    assert obs.current_email is not None
    assert obs.task_name == "easy_triage"
    assert obs.step_count == 0

def test_reset_sets_max_steps(easy_env):
    obs = easy_env.reset()
    assert obs.max_steps == TASKS["easy_triage"]["max_steps"]

def test_reset_inbox_summary(easy_env):
    obs = easy_env.reset()
    summary = obs.inbox_summary
    assert summary["total"] == 5
    assert summary["processed"] == 0
    assert summary["pending"] == 5

def test_reset_is_reproducible(easy_env):
    obs1 = easy_env.reset(seed=99)
    obs2 = easy_env.reset(seed=99)
    assert obs1.current_email["id"] == obs2.current_email["id"]

def test_reset_clears_state(easy_env):
    easy_env.reset(seed=1)
    # Take a step
    email_id = easy_env._emails[0]["id"]
    easy_env.step(EmailAction(email_id=email_id, action_type="label", priority="low"))
    # Reset and check clean state
    obs = easy_env.reset(seed=2)
    assert obs.step_count == 0
    assert easy_env._results == {}


# ── step() tests ────────────────────────────────────────────────────────────────

def test_step_label_action(easy_env):
    obs = easy_env.reset(seed=42)
    email = obs.current_email
    action = EmailAction(email_id=email["id"], action_type="label", priority="urgent")
    obs2 = easy_env.step(action)
    assert isinstance(obs2, EmailObservation)
    assert obs2.step_count == 1
    assert obs2.last_action_error is None

def test_step_invalid_action_type(easy_env):
    easy_env.reset()
    email_id = easy_env._emails[0]["id"]
    action = EmailAction(email_id=email_id, action_type="invalid_action")
    obs = easy_env.step(action)
    assert obs.last_action_error is not None
    assert obs.reward < 0

def test_step_label_without_priority_error(easy_env):
    easy_env.reset()
    email_id = easy_env._emails[0]["id"]
    action = EmailAction(email_id=email_id, action_type="label")  # missing priority
    obs = easy_env.step(action)
    assert obs.last_action_error is not None

def test_step_invalid_email_id(easy_env):
    easy_env.reset()
    action = EmailAction(email_id="nonexistent_id", action_type="archive")
    obs = easy_env.step(action)
    assert obs.last_action_error is not None

def test_step_correct_label_gives_positive_reward(easy_env):
    easy_env.reset(seed=42)
    # Find an email and label it correctly
    email = easy_env._emails[0]
    action = EmailAction(
        email_id=email["id"],
        action_type="label",
        priority=email["true_priority"],
    )
    obs = easy_env.step(action)
    assert obs.reward > 0

def test_step_wrong_label_gives_lower_reward(easy_env):
    easy_env.reset(seed=42)
    email = easy_env._emails[0]
    # Choose the opposite priority
    true_p = email["true_priority"]
    wrong_p = {"urgent": "low", "high": "low", "medium": "urgent", "low": "urgent"}[true_p]
    action = EmailAction(email_id=email["id"], action_type="label", priority=wrong_p)
    obs_wrong = easy_env.step(action)

    # Reset and try correct
    easy_env.reset(seed=42)
    action_correct = EmailAction(
        email_id=easy_env._emails[0]["id"],
        action_type="label",
        priority=easy_env._emails[0]["true_priority"],
    )
    obs_correct = easy_env.step(action_correct)
    assert obs_correct.reward >= obs_wrong.reward

def test_step_updates_score(easy_env):
    obs = easy_env.reset(seed=42)
    assert obs.score_so_far == 0.0
    email = easy_env._emails[0]
    action = EmailAction(
        email_id=email["id"],
        action_type="label",
        priority=email["true_priority"],
    )
    obs2 = easy_env.step(action)
    assert obs2.score_so_far >= 0.0

def test_episode_terminates_on_all_processed(easy_env):
    easy_env.reset(seed=42)
    # Process all emails
    for email in easy_env._emails:
        action = EmailAction(
            email_id=email["id"],
            action_type="label",
            priority=email["true_priority"],
        )
        obs = easy_env.step(action)
    assert obs.done is True

def test_episode_terminates_on_max_steps(easy_env):
    easy_env.reset(seed=42)
    max_steps = TASKS["easy_triage"]["max_steps"]
    obs = None
    for _ in range(max_steps):
        # Keep acting on the same email (won't advance, will hit step limit)
        email = easy_env._emails[0]
        action = EmailAction(email_id=email["id"], action_type="skip")
        obs = easy_env.step(action)
        if obs.done:
            break
    assert obs.done is True


# ── state() tests ───────────────────────────────────────────────────────────────

def test_state_returns_state(easy_env):
    easy_env.reset(seed=1)
    state = easy_env.state
    assert state.task_name == "easy_triage"
    assert state.step_count == 0
    assert isinstance(state.emails, list)

def test_state_tracks_results(easy_env):
    easy_env.reset(seed=1)
    email = easy_env._emails[0]
    easy_env.step(EmailAction(email_id=email["id"], action_type="label", priority="urgent"))
    state = easy_env.state
    assert email["id"] in state.results


# ── Reply quality scorer tests ──────────────────────────────────────────────────

def test_reply_quality_empty_is_zero():
    email = EMAIL_CORPUS[0]  # urgent email
    assert _score_reply_quality(email, "") == 0.0
    assert _score_reply_quality(email, "ok") == 0.0

def test_reply_quality_professional_reply():
    email = EMAIL_CORPUS[0]  # production outage email
    reply = (
        "Hi team, I understand this is urgent — production is down and we're losing revenue. "
        "I'm escalating this immediately to our on-call engineer. "
        "We will resolve this as soon as possible. "
        "Best regards, Support Team"
    )
    score = _score_reply_quality(email, reply)
    assert score >= 0.7

def test_reply_quality_returns_float_in_range():
    for email in EMAIL_CORPUS[:5]:
        score = _score_reply_quality(email, "Hello, thank you for reaching out.")
        assert 0.0 <= score <= 1.0


# ── Grader tests ────────────────────────────────────────────────────────────────

def test_grade_perfect_easy():
    task_cfg = TASKS["easy_triage"]
    email_ids = task_cfg["email_ids"]
    corpus_map = {e["id"]: e for e in EMAIL_CORPUS}
    emails = [corpus_map[eid] for eid in email_ids]

    # Perfect results
    results = {e["id"]: {"priority": e["true_priority"]} for e in emails}
    score = _grade_episode(task_cfg, emails, results)
    assert score == 1.0

def test_grade_empty_results_is_zero():
    task_cfg = TASKS["easy_triage"]
    email_ids = task_cfg["email_ids"]
    corpus_map = {e["id"]: e for e in EMAIL_CORPUS}
    emails = [corpus_map[eid] for eid in email_ids]
    score = _grade_episode(task_cfg, emails, {})
    assert score == 0.0

def test_grade_score_in_range():
    for task_name, task_cfg in TASKS.items():
        email_ids = task_cfg["email_ids"]
        corpus_map = {e["id"]: e for e in EMAIL_CORPUS}
        emails = [corpus_map[eid] for eid in email_ids]
        results = {e["id"]: {"priority": "medium", "action": "archive"} for e in emails}
        score = _grade_episode(task_cfg, emails, results)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for task {task_name}"

def test_grade_medium_perfect():
    task_cfg = TASKS["medium_triage"]
    email_ids = task_cfg["email_ids"]
    corpus_map = {e["id"]: e for e in EMAIL_CORPUS}
    emails = [corpus_map[eid] for eid in email_ids]
    results = {
        e["id"]: {"priority": e["true_priority"], "action": e["true_action"]}
        for e in emails
    }
    score = _grade_episode(task_cfg, emails, results)
    assert score >= 0.95  # near-perfect (reply quality not scored in medium)


# ── Full episode integration test ───────────────────────────────────────────────

def test_full_easy_episode_perfect_score():
    env = EmailTriageEnvironment(task_name="easy_triage")
    env.reset(seed=0)
    for email in env._emails:
        obs = env.step(EmailAction(
            email_id=email["id"],
            action_type="label",
            priority=email["true_priority"],
        ))
    assert obs.score_so_far == 1.0
    assert obs.done is True

def test_full_medium_episode_completes():
    env = EmailTriageEnvironment(task_name="medium_triage")
    env.reset(seed=42)
    for email in env._emails:
        # Label first
        env.step(EmailAction(
            email_id=email["id"],
            action_type="label",
            priority=email["true_priority"],
        ))
        # Then act
        obs = env.step(EmailAction(
            email_id=email["id"],
            action_type=email["true_action"],
        ))
    assert obs.done is True
    assert obs.score_so_far > 0.8

def test_task_switch_on_reset():
    env = EmailTriageEnvironment(task_name="easy_triage")
    obs = env.reset(task_name="hard_triage")
    assert obs.task_name == "hard_triage"
    assert obs.max_steps == TASKS["hard_triage"]["max_steps"]