"""
Inference Script — Email Triage OpenEnv
========================================
MANDATORY ENVIRONMENT VARIABLES:
  API_BASE_URL   LLM endpoint (default: HuggingFace router)
  MODEL_NAME     Model identifier
  HF_TOKEN       HuggingFace / API key
  IMAGE_NAME     Docker image name (for from_docker_image())

OPTIONAL:
  EMAIL_TRIAGE_TASK   Comma-separated task names to run
                      (default: easy_triage,medium_triage,hard_triage)

STDOUT FORMAT (strictly followed):
  [START] task=<n> env=email-triage-env model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# ── Config ─────────────────────────────────────────────────────────────────────
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "email-triage-env")
API_KEY          = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK        = "email-triage-env"
SUCCESS_THRESHOLD = 0.5

_RAW_TASKS = os.getenv("EMAIL_TRIAGE_TASK", "easy_triage,medium_triage,hard_triage")
TASK_NAMES = [t.strip() for t in _RAW_TASKS.split(",") if t.strip()]

TASK_MAX_STEPS = {
    "easy_triage":   10,
    "medium_triage": 16,
    "hard_triage":   30,
}

# ── System prompt ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert email triage assistant. You will process emails one at a time.

For each email, output ONLY a single JSON object — no markdown fences, no explanation:
{
  "email_id": "<exact id from current_email.id>",
  "action_type": "<label|reply|archive|delete|escalate>",
  "priority": "<urgent|high|medium|low>",
  "reply_body": "<draft reply text, or null>"
}

PRIORITY RULES:
- urgent : production down, security breach, legal deadline TODAY, revenue impact NOW
- high   : customer-facing bug, contract expiring this week, time-sensitive business
- medium : general business, non-urgent partner requests, HR/admin notices
- low    : newsletters, social notifications, automated system emails, routine billing

ACTION RULES:
- escalate : urgent/critical issues needing immediate human attention
- reply    : customer support questions, partner outreach, press requests, business asks
- archive  : informational emails, completed notifications, billing records, HR notices
- delete   : newsletters, social media pings, spam, irrelevant automated alerts
- label    : use ONLY for easy_triage task where only priority labeling is requested

REPLY QUALITY:
- Start with a professional greeting
- Reference the specific issue or request from the email
- For urgent issues: acknowledge urgency, state immediate next steps
- End with a professional sign-off
- Keep replies concise but complete (3-6 sentences)

Output ONLY valid JSON. No prose before or after.
""").strip()


# ── Logging helpers ─────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM interaction ─────────────────────────────────────────────────────────────
def build_user_prompt(obs_dict: Dict[str, Any], step: int, task_name: str) -> str:
    email   = obs_dict.get("current_email") or {}
    summary = obs_dict.get("inbox_summary", {})
    score   = obs_dict.get("score_so_far", 0.0)
    error   = obs_dict.get("last_action_error")

    easy_hint = ""
    if task_name == "easy_triage":
        easy_hint = "\nNOTE: This is easy_triage — use action_type='label' and set priority only.\n"

    error_block = ""
    if error:
        error_block = f"\n[!] Last action error: {error}\nPlease fix your JSON.\n"

    return textwrap.dedent(f"""
        Step: {step}
        Task: {task_name}
        Score so far: {score:.3f}
        {easy_hint}{error_block}
        ── Current Email ──────────────────────────────────
        ID      : {email.get('id', 'N/A')}
        From    : {email.get('from', 'N/A')}
        To      : {email.get('to', 'N/A')}
        Subject : {email.get('subject', 'N/A')}
        Body    : {email.get('body', 'N/A')}
        Has att : {email.get('has_attachment', False)}
        Labels  : {email.get('labels', [])}
        ────────────────────────────────────────────────────
        Inbox  : {summary.get('processed', 0)}/{summary.get('total', 0)} processed
                  {summary.get('pending', 0)} remaining

        Output your JSON action now.
    """).strip()


def get_model_action(
    client: OpenAI,
    obs_dict: Dict[str, Any],
    step: int,
    task_name: str,
    history: List[str],
) -> Dict[str, Any]:
    """Call the LLM and return parsed action dict."""
    user_prompt = build_user_prompt(obs_dict, step, task_name)

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for h in history[-4:]:
        messages.append({"role": "assistant", "content": h})
    messages.append({"role": "user", "content": user_prompt})

    raw = ""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed: Dict[str, Any] = json.loads(raw)
        return parsed

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e} — raw: {raw[:200]!r}", flush=True)
        email_id = (obs_dict.get("current_email") or {}).get("id", "")
        return {
            "email_id": email_id,
            "action_type": "archive",
            "priority": "low",
            "reply_body": None,
        }
    except Exception as e:
        print(f"[DEBUG] LLM request failed: {e}", flush=True)
        email_id = (obs_dict.get("current_email") or {}).get("id", "")
        return {
            "email_id": email_id,
            "action_type": "archive",
            "priority": "low",
            "reply_body": None,
        }


# ── Single task runner ──────────────────────────────────────────────────────────
async def run_task(task_name: str) -> None:
    """Run one complete episode for the given task."""
    from client import EmailTriageEnv
    from models import EmailAction

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    max_steps   = TASK_MAX_STEPS.get(task_name, 20)
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    history: List[str] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = await EmailTriageEnv.from_docker_image(
        LOCAL_IMAGE_NAME,
        env_vars={"EMAIL_TRIAGE_TASK": task_name},
    )

    try:
        result   = await env.reset(task_name=task_name)
        obs      = result.observation
        obs_dict = obs.model_dump()

        for step in range(1, max_steps + 1):
            if result.done:
                break

            current_email = obs_dict.get("current_email")
            if current_email is None:
                print(f"[DEBUG] No more emails to process at step {step}", flush=True)
                break

            action_dict = get_model_action(client, obs_dict, step, task_name, history)

            email_id    = action_dict.get("email_id", "")
            action_type = action_dict.get("action_type", "archive")
            priority    = action_dict.get("priority")
            reply_body  = action_dict.get("reply_body")

            if not email_id:
                email_id = current_email.get("id", "")

            try:
                action_kwargs: Dict[str, Any] = {
                    "email_id": email_id,
                    "action_type": action_type,
                }
                if priority:
                    action_kwargs["priority"] = priority
                if reply_body:
                    action_kwargs["reply_body"] = reply_body

                action = EmailAction(**action_kwargs)

            except Exception as e:
                log_step(step, "invalid_action", 0.0, False, str(e)[:120])
                rewards.append(0.0)
                steps_taken = step
                continue

            result   = await env.step(action)
            obs      = result.observation
            obs_dict = obs.model_dump()

            reward = float(result.reward or 0.0)
            done   = result.done
            error  = obs.last_action_error

            rewards.append(reward)
            steps_taken = step

            action_label = f"{action_type}({email_id},priority={priority or '-'})"
            log_step(
                step=step,
                action=action_label,
                reward=reward,
                done=done,
                error=error,
            )

            history.append(json.dumps(action_dict))

            if done:
                break

        score   = float(obs.score_so_far or 0.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


# ── Entry point ─────────────────────────────────────────────────────────────────
async def main() -> None:
    for task_name in TASK_NAMES:
        await run_task(task_name)


if __name__ == "__main__":
    asyncio.run(main())