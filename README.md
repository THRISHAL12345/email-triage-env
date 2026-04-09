# 📬 Email Triage OpenEnv

A real-world **email inbox triage environment** for training and evaluating AI agents.  
Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) spec by Meta & Hugging Face.

---

## Why Email Triage?

Email triage is one of the most common, high-value knowledge-worker tasks:

- Knowledge workers spend **28% of their workweek** managing email (McKinsey)
- Enterprise AI assistants must handle **priority classification**, **action routing**, and **professional reply drafting**
- Unlike games or coding tasks, email triage tests **real-world language understanding** under ambiguity

This environment fills a gap in the OpenEnv ecosystem: no existing environment covers natural language inbox management with graded reward signals across all three dimensions.

---

## Environment Overview

| Property | Value |
|----------|-------|
| Task type | Classification + Action selection + NLG |
| Episodes | Per-task inbox of 5–12 emails |
| Reward type | Dense (per-step partial credit) |
| Tasks | 3 (easy → medium → hard) |
| Action space | Structured JSON (label/reply/archive/delete/escalate) |
| Observation space | Email content + inbox state + running score |

---

## Tasks

### 🟢 `easy_triage` — Priority Labeling
- **5 emails** with clear, unambiguous priority signals
- Agent must assign: `urgent | high | medium | low`
- Max 10 steps
- Expected score range: 0.7–1.0 for capable models

### 🟡 `medium_triage` — Priority + Action Selection
- **8 emails** with mixed signals and some ambiguity
- Agent must assign priority AND choose: `reply | archive | delete | escalate`
- Max 16 steps  
- Expected score range: 0.45–0.75 for frontier models

### 🔴 `hard_triage` — Full Triage with Reply Drafting
- **12 emails** across all categories
- Agent must label priority, choose action, AND draft relevant professional replies for emails that require a response
- Reply quality scored on: relevance, professionalism, urgency acknowledgment
- Max 30 steps
- Expected score range: 0.30–0.60 for frontier models

---

## Action Space

```json
{
  "email_id": "e001",
  "action_type": "label",
  "priority": "urgent",
  "reply_body": null
}
```

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `email_id` | string | ✅ | ID of target email |
| `action_type` | string | ✅ | `label \| reply \| archive \| delete \| escalate \| skip` |
| `priority` | string | when `action_type=label` | `urgent \| high \| medium \| low` |
| `reply_body` | string | when `action_type=reply` | Full reply text |

**Action semantics:**
- `label` — assign a priority level to the email
- `reply` — compose and send a reply (requires `reply_body`)
- `archive` — file the email, no response needed
- `delete` — remove from inbox (spam, newsletters, irrelevant)
- `escalate` — flag as requiring immediate human attention
- `skip` — pass (penalized: −0.05 reward)

---

## Observation Space

```json
{
  "current_email": {
    "id": "e001",
    "from": "cto@enterprise.com",
    "subject": "URGENT: Production database is down",
    "body": "...",
    "timestamp": "2024-03-15T09:02:00Z",
    "has_attachment": false,
    "labels": []
  },
  "inbox_summary": {
    "total": 5,
    "processed": 1,
    "pending": 4,
    "current_index": 1
  },
  "last_action_result": "Action 'label' recorded for email 'e001'. Progress: 1/5.",
  "last_action_error": null,
  "task_name": "easy_triage",
  "step_count": 1,
  "max_steps": 10,
  "score_so_far": 0.4,
  "done": false,
  "reward": 0.4
}
```

---

## Reward Function

Rewards are **dense** — the agent receives signal at every step, not just at episode end.

```
step_reward = (
    priority_accuracy  × task_priority_weight  × 0.4
  + action_accuracy    × task_action_weight    × 0.4
  + reply_quality      × task_reply_weight     × 0.4
  − 0.05 × (action == "skip")
  − 0.10 × invalid_action
)
```

**Priority accuracy:** Full credit for exact match, partial credit for adjacent levels (e.g., predicting `high` when true is `urgent` gets 0.65 credit).

**Action accuracy:** Full credit for exact match, partial credit for semantically similar choices (e.g., `reply` when true is `escalate` gets 0.5 credit).

**Reply quality (hard task):** Heuristic scorer measuring:
- Message length and substance (0.2)
- Topic relevance — keyword overlap with subject/body (0.3)
- Professional tone — greeting + sign-off (0.2)
- Urgency acknowledgment for high-priority emails (0.3)

**Task weights:**

| Task | Priority | Action | Reply |
|------|---------|--------|-------|
| easy | 1.0 | 0.0 | 0.0 |
| medium | 0.5 | 0.5 | 0.0 |
| hard | 0.35 | 0.35 | 0.30 |

**Final episode score** = weighted average across all emails (0.0–1.0).

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker
- `pip install openenv-core`

### Local Development

```bash
# Clone and install
git clone https://huggingface.co/spaces/your-org/email-triage-env
cd email-triage-env
pip install -e .

# Run server locally
PYTHONPATH=. uvicorn email_triage_env.server.app:app --host 0.0.0.0 --port 7860

# In another terminal — quick smoke test
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -m json.tool
```

### Docker

```bash
# Build (from repo root)
docker build -f email_triage_env/server/Dockerfile -t email-triage-env .

# Run easy task
docker run -p 7860:7860 -e EMAIL_TRIAGE_TASK=easy_triage email-triage-env

# Run hard task
docker run -p 7860:7860 -e EMAIL_TRIAGE_TASK=hard_triage email-triage-env

# Test the server
curl -s http://localhost:7860/health
curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

### Running the Baseline Inference Script

```bash
export HF_TOKEN=hf_your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export IMAGE_NAME=email-triage-env   # local Docker image
export EMAIL_TRIAGE_TASK=easy_triage,medium_triage,hard_triage

python inference.py
```

Expected output:
```
[START] task=easy_triage env=email-triage-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=label(e001,priority=urgent) reward=0.40 done=false error=null
[STEP] step=2 action=label(e010,priority=low) reward=0.40 done=false error=null
...
[END] success=true steps=5 score=0.820 rewards=0.40,0.40,0.40,0.40,0.00
```

### OpenEnv Validation

```bash
openenv validate
```

### Deploy to Hugging Face Spaces

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Push
openenv push --repo-id your-org/email-triage-env
```

---

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Score | Notes |
|------|-------|-------|
| `easy_triage` | ~0.82 | Strong on clear signals, occasional priority confusion |
| `medium_triage` | ~0.61 | Action selection harder; reply/escalate confusion |
| `hard_triage` | ~0.47 | Reply quality limits score; urgency acknowledgment weak |

**Frontier model ceiling** (GPT-4o, Claude 3.5):

| Task | Score |
|------|-------|
| `easy_triage` | ~0.95 |
| `medium_triage` | ~0.78 |
| `hard_triage` | ~0.65 |

The hard task remains genuinely challenging: drafting relevant, professional replies that reference specific email context requires deep language understanding beyond surface-level classification.

---

## Email Corpus

The environment includes **15 realistic email scenarios** across 9 categories:

| Category | Examples |
|----------|---------|
| Incident | Production DB down, security alert |
| Legal | Contract deadline, NDAs |
| Sales | Enterprise renewal, pricing request |
| Support | Bug reports, how-to questions |
| HR | Benefits enrollment, expense reports |
| Business | Partnership proposals, press requests |
| Billing | AWS invoices, subscription notices |
| Notifications | GitHub PR approvals, LinkedIn pings |
| Newsletter | Marketing, social media digests |

Each email has ground-truth labels for: `true_priority`, `true_category`, `true_action`, and `requires_response`.

---

## Project Structure

```
email-triage-env/
├── __init__.py          # EmailTriageEnv, EmailAction, EmailObservation
├── models.py            # Pydantic model exports
├── client.py            # WebSocket EnvClient
├── inference.py         # ← Baseline agent (root level, per OpenEnv spec)
├── openenv.yaml         # Environment manifest
├── pyproject.toml       # Package config
├── README.md
└── server/
    ├── __init__.py
    ├── email_data.py    # 15 email scenarios + 3 task configs
    ├── environment.py   # EmailTriageEnvironment core logic
    ├── app.py           # FastAPI via openenv create_app()
    ├── requirements.txt
    └── Dockerfile
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMAIL_TRIAGE_TASK` | `easy_triage` | Active task name |
| `ENABLE_WEB_INTERFACE` | `false` | Enable Gradio UI at `/web` |
| `PORT` | `7860` | Server port |
| `API_BASE_URL` | HF router | LLM endpoint for inference |
| `MODEL_NAME` | Qwen2.5-72B | Model for inference script |
| `HF_TOKEN` | — | HuggingFace API key |
| `IMAGE_NAME` | — | Docker image for client |

---

## License

MIT — contributions welcome.