"""
Grader for easy_triage task
=============================
Scores based on priority label accuracy. Each email contributes equally.
Perfect priority match: 0.2 points. Partial credit for close matches.
"""


from typing import Any, Dict




PRIORITY_LEVELS = ["urgent", "high", "medium", "low"]




def grade(results: Dict[str, Any], emails: list[Dict[str, Any]]) -> float:
    """
    Grade the easy_triage task based on priority accuracy.
   
    Args:
        results: Dictionary mapping email_id to agent's decisions
        emails: List of email dictionaries with ground truth
   
    Returns:
        Score between 0.0 and 1.0
    """
    if not emails:
        return 0.0
   
    email_map = {e["id"]: e for e in emails}
    priority_scores = []
   
    for email in emails:
        eid = email["id"]
        result = results.get(eid, {})
       
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
   
    # Return average priority score
    return round(sum(priority_scores) / len(priority_scores), 4) if priority_scores else 0.0