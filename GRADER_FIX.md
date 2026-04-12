# Grader Implementation Fix

## Problem
The OpenEnv validator was failing with:
```
❌ Not enough tasks with graders
Your submission must include at least 3 tasks with graders.
```

## Root Cause
While the `openenv.yaml` defined 3 tasks (easy_triage, medium_triage, hard_triage) with grader configurations, and the environment had internal grading logic via `_grade_episode()`, the OpenEnv framework requires an explicit public `grade()` method in the Environment class.

## Solution
Added a public `grade()` method to the `EmailTriageEnvironment` class in `server/environment.py`:

```python
def grade(self) -> float:
    """
    Grade the current episode and return a score in [0.0, 1.0].
    
    This method is called by the OpenEnv framework to compute the final
    score for an episode based on the agent's performance.
    """
    return _grade_episode(self._task_cfg, self._emails, self._results)
```

## Verification
All 3 tasks now have working graders:

### 1. easy_triage
- **Grader type**: programmatic (environment method)
- **Scoring**: Priority label accuracy (100% weight)
- **Max steps**: 10
- **Score range**: [0.0, 1.0]

### 2. medium_triage
- **Grader type**: programmatic (environment method)
- **Scoring**: Priority accuracy (50% weight) + Action correctness (50% weight)
- **Max steps**: 16
- **Score range**: [0.0, 1.0]

### 3. hard_triage
- **Grader type**: programmatic (environment method)
- **Scoring**: Priority (35%) + Action (35%) + Reply quality (30%)
- **Max steps**: 30
- **Score range**: [0.0, 1.0]

## What Changed
- **File modified**: `server/environment.py`
- **Change**: Added public `grade()` method (lines ~497-505)
- **Impact**: OpenEnv validator can now detect all 3 tasks have functional graders

## Next Steps
Resubmit your environment to the OpenEnv validator. The "Not enough tasks with graders" error should now be resolved.
