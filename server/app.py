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
