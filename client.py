"""
Email Triage Environment — Client
"""

from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.env_client import StepResult
from models import EmailAction, EmailObservation, EmailState


class EmailTriageEnv(EnvClient[EmailAction, EmailObservation, EmailState]):
    """Client for the Email Triage Environment."""

    action_cls = EmailAction
    observation_cls = EmailObservation

    def _step_payload(self, action: EmailAction) -> Dict[str, Any]:
        """Convert action to JSON payload for the server."""
        data = {
            "email_id": action.email_id,
            "action_type": action.action_type,
        }
        if action.priority is not None:
            data["priority"] = action.priority
        if action.reply_body is not None:
            data["reply_body"] = action.reply_body
        return data

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[EmailObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", payload)
        obs = EmailObservation(**obs_data)
        # Get reward from top-level payload first, then fall back to observation
        reward = payload.get("reward", obs.reward) or 0.0
        done = payload.get("done", obs.done)
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> EmailState:
        """Parse server state response into EmailState."""
        return EmailState(**payload)