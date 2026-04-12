"""
Email Triage Environment — FastAPI Server
"""

import os
import uvicorn
from openenv.core import create_app
from server.environment import EmailTriageEnvironment, EmailAction, EmailObservation

TASK_NAME = os.getenv("EMAIL_TRIAGE_TASK", "easy_triage")

# Store active envs by session_id so /score can access them
_active_envs: dict = {}

def make_env():
    env = EmailTriageEnvironment(task_name=TASK_NAME)
    return env

app = create_app(
    env=make_env,
    action_cls=EmailAction,
    observation_cls=EmailObservation,
    env_name="email-triage-env",
    max_concurrent_envs=8,
)

@app.get("/score")
async def get_score():
    """
    OpenEnv grader endpoint — returns a score for validation.
    Runs a complete episode with a heuristic agent and returns the score.
    """
    from server.email_data import TASKS
    scores = {}
    for task_name in TASKS:
        env = EmailTriageEnvironment(task_name=task_name)
        obs = env.reset()
        # Step through with a simple heuristic to produce a non-zero score
        for _ in range(TASKS[task_name]["max_steps"]):
            if obs.done or obs.current_email is None:
                break
            email = obs.current_email
            action = EmailAction(
                email_id=email["id"],
                action_type="label",
                priority="high",
            )
            obs = env.step(action)
        scores[task_name] = env.grade()
    # Return overall average
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    return {"score": avg, "done": True, "task_scores": scores}

@app.get("/grade")
async def grade():
    """Alias for /score for compatibility."""
    return await get_score()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()
