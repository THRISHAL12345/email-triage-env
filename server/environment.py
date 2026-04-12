"""
Email Triage Environment — Core Logic
======================================
Implements the EmailTriageEnvironment with:
  - Full OpenEnv spec (step / reset / state)
  - Three tasks: easy_triage, medium_triage, hard_triage
  - Dense reward function with partial credit
  - Reply-quality scoring via keyword heuristics
"""

from __future__ import annotations

import json
import random
import re
import uuid
from typing import Any, Dict, List, Optional

from openenv.core import Action, Environment, Observation, State

from .email_data import EMAIL_CORPUS, TASKS, PRIORITY_LEVELS, ACTION_TYPES


# ──────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ──────────────────────────────────────────────────────────────

class EmailAction(Action):
    """Action taken on a single email in the inbox."""

    email_id: str
    """ID of the email this action targets (e.g. 'e001')."""

    action_type: str
    """One of: label | reply | archive | delete | escalate | skip."""

    priority: Optional[str] = None
    """Priority label: urgent | high | medium | low. Required for 'label' actions."""

    reply_body: Optional[str] = None
    """Draft reply text. Required when action_type == 'reply'."""


class EmailObservation(Observation):
    """Observation returned after each step."""

    current_email: Optional[Dict[str, Any]] = None
    """The email currently being shown to the agent."""

    inbox_summary: Dict[str, Any] = {}
    """Summary of inbox: total, processed, pending counts."""

    last_action_result: str = ""
    """Human-readable result of the last action taken."""

    last_action_error: Optional[str] = None
    """Error message if the last action was invalid."""

    task_name: str = ""
    """Name of the active task."""

    task_description: str = ""
    """Description of the task."""

    step_count: int = 0
    """Current step number."""

    max_steps: int = 0
    """Maximum steps allowed."""

    score_so_far: float = 0.0
    """Cumulative score in [0, 1] as of this step."""


class EmailState(State):
    """Internal environment state."""

    task_name: str = ""
    emails: List[Dict[str, Any]] = []
    current_index: int = 0
    results: Dict[str, Dict[str, Any]] = {}  # email_id -> {priority, action, reply}
    score_so_far: float = 0.0


# ──────────────────────────────────────────────────────────────
# REPLY QUALITY SCORER
# ──────────────────────────────────────────────────────────────

def _score_reply_quality(email: Dict[str, Any], reply: str) -> float:
    """
    Heuristic scorer for reply quality. Returns 0.0 – 1.0.
    Checks: non-empty, relevant keywords, professional tone, length.
    """
    if not reply or len(reply.strip()) < 10:
        return 0.0

    score = 0.0
    reply_lower = reply.lower()
    subject_words = set(re.findall(r'\w+', email["subject"].lower()))
    body_words = set(re.findall(r'\w+', email["body"].lower()))

    # 1. Non-empty meaningful reply (0.2)
    if len(reply.strip()) >= 20:
        score += 0.2

    # 2. References the email topic (0.3)
    relevant_words = (subject_words | body_words) - {
        "the", "a", "an", "is", "are", "was", "we", "i", "you", "your", "our",
        "this", "that", "to", "for", "and", "in", "of", "on", "at", "have"
    }
    matched = sum(1 for w in relevant_words if w in reply_lower)
    if matched >= 3:
        score += 0.3
    elif matched >= 1:
        score += 0.15

    # 3. Professional tone (0.2) — greeting + sign-off
    has_greeting = any(g in reply_lower for g in ["hi ", "hello ", "dear ", "thank"])
    has_signoff = any(s in reply_lower for s in ["regards", "sincerely", "best", "thanks", "thank you"])
    if has_greeting:
        score += 0.1
    if has_signoff:
        score += 0.1

    # 4. Urgency acknowledgment for urgent emails (0.3)
    if email["true_priority"] in ("urgent", "high"):
        urgency_words = ["immediately", "right away", "urgent", "priority", "asap",
                         "escalat", "on-call", "team", "investigate", "fix", "resolve"]
        if any(w in reply_lower for w in urgency_words):
            score += 0.3
        else:
            score += 0.1  # partial credit for replying at all
    else:
        # For non-urgent, just having substance is enough
        if len(reply.strip()) >= 50:
            score += 0.3
        elif len(reply.strip()) >= 20:
            score += 0.15

    return min(score, 1.0)


# ──────────────────────────────────────────────────────────────
# GRADER
# ──────────────────────────────────────────────────────────────

def _grade_episode(
    task_cfg: Dict[str, Any],
    emails: List[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
) -> float:
    """
    Final episode grader: returns 0.0 – 1.0 score.
    """
    cfg = task_cfg["grader_config"]
    priority_w = cfg["priority_weight"]
    action_w = cfg["action_weight"]
    reply_w = cfg["response_quality_weight"]

    email_map = {e["id"]: e for e in emails}
    n = len(emails)
    if n == 0:
        return 0.0

    priority_scores, action_scores, reply_scores = [], [], []

    for email in emails:
        eid = email["id"]
        result = results.get(eid, {})

        # Priority scoring
        if priority_w > 0:
            pred_priority = result.get("priority")
            true_priority = email["true_priority"]
            if pred_priority == true_priority:
                priority_scores.append(1.0)
            elif pred_priority is not None:
                # Partial credit for adjacent levels
                levels = PRIORITY_LEVELS
                if true_priority in levels and pred_priority in levels:
                    dist = abs(levels.index(pred_priority) - levels.index(true_priority))
                    priority_scores.append(max(0.0, 1.0 - dist * 0.35))
                else:
                    priority_scores.append(0.0)
            else:
                priority_scores.append(0.0)

        # Action scoring
        if action_w > 0:
            pred_action = result.get("action")
            true_action = email["true_action"]
            if pred_action == true_action:
                action_scores.append(1.0)
            elif pred_action is not None:
                # Partial credit: reply≈escalate, archive≈delete
                partial = {
                    ("reply", "escalate"): 0.5,
                    ("escalate", "reply"): 0.5,
                    ("archive", "delete"): 0.4,
                    ("delete", "archive"): 0.4,
                }
                action_scores.append(partial.get((pred_action, true_action), 0.0))
            else:
                action_scores.append(0.0)

        # Reply quality scoring
        if reply_w > 0 and email.get("requires_response"):
            reply_text = result.get("reply_body", "")
            reply_scores.append(_score_reply_quality(email, reply_text or ""))

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    total = 0.0
    if priority_w > 0:
        total += priority_w * safe_avg(priority_scores)
    if action_w > 0:
        total += action_w * safe_avg(action_scores)
    if reply_w > 0:
        responders = [e for e in emails if e.get("requires_response")]
        if responders:
            total += reply_w * safe_avg(reply_scores)

    return round(min(max(total, 0.0), 1.0), 4)


# ──────────────────────────────────────────────────────────────
# ENVIRONMENT
# ──────────────────────────────────────────────────────────────

class EmailTriageEnvironment(Environment[EmailAction, EmailObservation, EmailState]):
    """
    Email Triage Environment
    ========================
    Simulates a real-world email inbox triage workflow.
    The agent must classify email priority, choose the right action,
    and draft quality replies — key skills for AI assistants and RL agents.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_name: str = "easy_triage"):
        super().__init__()
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid tasks: {list(TASKS.keys())}"
            )
        self._task_name = task_name
        self._task_cfg = TASKS[task_name]

        # Episode state
        self._emails: List[Dict[str, Any]] = []
        self._current_index: int = 0
        self._results: Dict[str, Dict[str, Any]] = {}
        self._step_count: int = 0
        self._episode_id: str = ""
        self._done: bool = False
        self._score: float = 0.0

    # ── helpers ──────────────────────────────────────────────

    def _get_current_email(self) -> Optional[Dict[str, Any]]:
        if self._current_index < len(self._emails):
            return self._emails[self._current_index]
        return None

    def _build_inbox_summary(self) -> Dict[str, Any]:
        processed = len(self._results)
        total = len(self._emails)
        return {
            "total": total,
            "processed": processed,
            "pending": total - processed,
            "current_index": self._current_index,
        }

    def _compute_step_reward(
        self, email: Dict[str, Any], action: EmailAction
    ) -> float:
        """Dense per-step reward providing partial progress signal."""
        reward = 0.0
        cfg = self._task_cfg["grader_config"]

        # Priority correctness
        if action.action_type == "label" and action.priority:
            true_p = email["true_priority"]
            pred_p = action.priority
            if pred_p == true_p:
                reward += 0.4 * cfg["priority_weight"]
            elif pred_p in PRIORITY_LEVELS:
                levels = PRIORITY_LEVELS
                dist = abs(levels.index(pred_p) - levels.index(true_p))
                reward += max(0.0, 0.4 - dist * 0.15) * cfg["priority_weight"]

        # Action correctness
        if action.action_type in ACTION_TYPES and action.action_type != "label":
            true_a = email["true_action"]
            if action.action_type == true_a:
                reward += 0.4 * cfg["action_weight"]
            else:
                partial = {
                    ("reply", "escalate"): 0.2,
                    ("escalate", "reply"): 0.2,
                    ("archive", "delete"): 0.15,
                    ("delete", "archive"): 0.15,
                }
                reward += partial.get((action.action_type, true_a), 0.0) * cfg["action_weight"]

        # Reply quality
        if action.action_type == "reply" and action.reply_body:
            q = _score_reply_quality(email, action.reply_body)
            reward += q * 0.4 * cfg["response_quality_weight"]

        # Small penalty for skipping without good reason
        if action.action_type == "skip":
            reward -= 0.05

        return round(reward, 4)

    # ── OpenEnv API ──────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> EmailObservation:
        """Reset the environment for a new episode."""
        if task_name and task_name in TASKS:
            self._task_name = task_name
            self._task_cfg = TASKS[task_name]

        if seed is not None:
            random.seed(seed)

        # Load emails for this task
        email_ids = self._task_cfg["email_ids"]
        corpus_map = {e["id"]: e for e in EMAIL_CORPUS}
        self._emails = [corpus_map[eid] for eid in email_ids if eid in corpus_map]

        # Shuffle for variety (seeded)
        random.shuffle(self._emails)

        self._current_index = 0
        self._results = {}
        self._step_count = 0
        self._episode_id = episode_id or str(uuid.uuid4())
        self._done = False
        self._score = 0.0

        current = self._get_current_email()

        return EmailObservation(
            done=False,
            reward=0.0,
            current_email=current,
            inbox_summary=self._build_inbox_summary(),
            last_action_result="Episode started. Process the email shown in current_email.",
            task_name=self._task_name,
            task_description=self._task_cfg["description"],
            step_count=0,
            max_steps=self._task_cfg["max_steps"],
            score_so_far=0.0,
        )

    def step(
        self,
        action: EmailAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> EmailObservation:
        """Take one triage action on an email."""
        self._step_count += 1
        max_steps = self._task_cfg["max_steps"]

        # Validate action
        if action.action_type not in ACTION_TYPES:
            return EmailObservation(
                done=False,
                reward=-0.1,
                current_email=self._get_current_email(),
                inbox_summary=self._build_inbox_summary(),
                last_action_result="",
                last_action_error=f"Invalid action_type '{action.action_type}'. Must be one of {ACTION_TYPES}",
                task_name=self._task_name,
                task_description=self._task_cfg["description"],
                step_count=self._step_count,
                max_steps=max_steps,
                score_so_far=self._score,
            )

        if action.action_type == "label" and not action.priority:
            return EmailObservation(
                done=False,
                reward=-0.05,
                current_email=self._get_current_email(),
                inbox_summary=self._build_inbox_summary(),
                last_action_result="",
                last_action_error="'label' action requires a 'priority' field.",
                task_name=self._task_name,
                task_description=self._task_cfg["description"],
                step_count=self._step_count,
                max_steps=max_steps,
                score_so_far=self._score,
            )

        # Find target email
        email_map = {e["id"]: e for e in self._emails}
        target_email = email_map.get(action.email_id)

        if target_email is None:
            return EmailObservation(
                done=False,
                reward=-0.1,
                current_email=self._get_current_email(),
                inbox_summary=self._build_inbox_summary(),
                last_action_result="",
                last_action_error=f"Email ID '{action.email_id}' not found in this task's inbox.",
                task_name=self._task_name,
                task_description=self._task_cfg["description"],
                step_count=self._step_count,
                max_steps=max_steps,
                score_so_far=self._score,
            )

        # Record the result for this email
        if action.email_id not in self._results:
            self._results[action.email_id] = {}

        if action.action_type == "label" and action.priority:
            self._results[action.email_id]["priority"] = action.priority
        elif action.action_type != "skip":
            self._results[action.email_id]["action"] = action.action_type
            if action.priority:
                self._results[action.email_id]["priority"] = action.priority
            if action.reply_body:
                self._results[action.email_id]["reply_body"] = action.reply_body

        # Compute step reward
        step_reward = self._compute_step_reward(target_email, action)

        # Advance to next unprocessed email if current one is fully handled
        self._advance_if_done(action)

        # Update running score
        self._score = _grade_episode(self._task_cfg, self._emails, self._results)

        # Termination check
        all_processed = len(self._results) >= len(self._emails)
        step_limit = self._step_count >= max_steps
        done = all_processed or step_limit
        self._done = done

        result_msg = (
            f"Action '{action.action_type}' recorded for email '{action.email_id}'. "
            f"Progress: {len(self._results)}/{len(self._emails)} emails processed."
        )
        if done:
            result_msg += f" Episode complete! Final score: {self._score:.3f}"

        next_email = self._get_current_email()

        return EmailObservation(
            done=done,
            reward=step_reward,
            current_email=next_email,
            inbox_summary=self._build_inbox_summary(),
            last_action_result=result_msg,
            last_action_error=None,
            task_name=self._task_name,
            task_description=self._task_cfg["description"],
            step_count=self._step_count,
            max_steps=max_steps,
            score_so_far=self._score,
        )

    def _advance_if_done(self, action: EmailAction) -> None:
        """Move index forward if the current email has been sufficiently processed."""
        current = self._get_current_email()
        if current is None:
            return
        if current["id"] == action.email_id:
            result = self._results.get(action.email_id, {})
            required = self._task_cfg["required_actions"]
            # Consider email processed if at least one required action type is recorded
            has_priority = "priority" in result
            has_action = "action" in result
            if "label" in required and "take_action" in required:
                if has_priority and has_action:
                    self._current_index += 1
            elif "label" in required:
                if has_priority:
                    self._current_index += 1
            else:
                if has_action:
                    self._current_index += 1

    def grade(self) -> float:
        """
        Grade the current episode and return a score in [0.0, 1.0].
        
        This method is called by the OpenEnv framework to compute the final
        score for an episode based on the agent's performance.
        """
        return _grade_episode(self._task_cfg, self._emails, self._results)

    @property
    def state(self) -> EmailState:
        return EmailState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            emails=self._emails,
            current_index=self._current_index,
            results=self._results,
            score_so_far=self._score,
        )
