"""
Grader for hard_triage task
============================
Comprehensive grading: priority (35%), action (35%), reply quality (30%).
Reply quality assessed via keyword matching and professional tone indicators.
"""


import re
from typing import Any, Dict




PRIORITY_LEVELS = ["urgent", "high", "medium", "low"]




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




def grade(results: Dict[str, Any], emails: list[Dict[str, Any]]) -> float:
    """
    Grade the hard_triage task based on priority, action, and reply quality.
   
    Args:
        results: Dictionary mapping email_id to agent's decisions
        emails: List of email dictionaries with ground truth
   
    Returns:
        Score between 0.0 and 1.0
    """
    if not emails:
        return 0.0
   
    priority_weight = 0.35
    action_weight = 0.35
    reply_weight = 0.30
   
    priority_scores = []
    action_scores = []
    reply_scores = []
   
    for email in emails:
        eid = email["id"]
        result = results.get(eid, {})
       
        # Priority scoring
        pred_priority = result.get("priority")
        true_priority = email["true_priority"]
       
        if pred_priority == true_priority:
            priority_scores.append(1.0)
        elif pred_priority is not None:
            # Partial credit for adjacent levels
            if true_priority in PRIORITY_LEVELS and pred_priority in PRIORITY_LEVELS:
                dist = abs(PRIORITY_LEVELS.index(pred_priority) - PRIORITY_LEVELS.index(true_priority))
                priority_scores.append(max(0.0, 1.0 - dist * 0.35))
            else:
                priority_scores.append(0.0)
        else:
            priority_scores.append(0.0)
       
        # Action scoring
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
       
        # Reply quality scoring (only for emails that require responses)
        if email.get("requires_response"):
            reply_text = result.get("reply_body", "")
            reply_scores.append(_score_reply_quality(email, reply_text or ""))
   
    # Calculate weighted average
    priority_avg = sum(priority_scores) / len(priority_scores) if priority_scores else 0.0
    action_avg = sum(action_scores) / len(action_scores) if action_scores else 0.0
   
    total = priority_weight * priority_avg + action_weight * action_avg
   
    # Add reply quality component if applicable
    responders = [e for e in emails if e.get("requires_response")]
    if responders and reply_scores:
        reply_avg = sum(reply_scores) / len(reply_scores)
        total += reply_weight * reply_avg
   
    return round(min(max(total, 0.0), 1.0), 4)