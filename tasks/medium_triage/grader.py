"""
Grader for medium_triage task
===============================
Scores based on priority accuracy (60%) and action correctness (40%).
Weighted average across all processed emails.
"""

from typing import Any, Dict


PRIORITY_LEVELS = ["urgent", "high", "medium", "low"]


def grade(results: Dict[str, Any], emails: list[Dict[str, Any]]) -> float:
    """
    Grade the medium_triage task based on priority and action accuracy.
    
    Args:
        results: Dictionary mapping email_id to agent's decisions
        emails: List of email dictionaries with ground truth
    
    Returns:
        Score between 0.0 and 1.0
    """
    if not emails:
        return 0.0
    
    priority_weight = 0.6
    action_weight = 0.4
    
    priority_scores = []
    action_scores = []
    
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
    
    # Calculate weighted average
    priority_avg = sum(priority_scores) / len(priority_scores) if priority_scores else 0.0
    action_avg = sum(action_scores) / len(action_scores) if action_scores else 0.0
    
    total = priority_weight * priority_avg + action_weight * action_avg
    
    return round(min(max(total, 0.0), 1.0), 4)
