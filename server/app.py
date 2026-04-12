"""
Email Triage Environment — FastAPI Server
==========================================
Serves the environment via OpenEnv's create_app() convention.
"""

import os
import uvicorn
from openenv.core import create_app
from server.environment import EmailTriageEnvironment, EmailAction, EmailObservation

TASK_NAME = os.getenv("EMAIL_TRIAGE_TASK", "easy_triage")


def make_env():
    return EmailTriageEnvironment(task_name=TASK_NAME)


app = create_app(
    env=make_env,
    action_cls=EmailAction,
    observation_cls=EmailObservation,
    env_name="email-triage-env",
    max_concurrent_envs=8,
)


@app.get("/score")
async def get_score(session_id: str | None = None):
    """
    Get the final score for a session.
    
    This endpoint is used by the OpenEnv grader to retrieve the score
    for an episode after it has been completed.
    """
    from openenv.core import get_session
    
    env = get_session(session_id)
    if env is None:
        return {"error": "Session not found", "score": 0.0, "done": False}
    
    score = env.grade()
    done = env._done if hasattr(env, '_done') else False
    
    return {"score": score, "done": done}


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
