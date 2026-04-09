"""
Email Triage Environment — Email Dataset & Task Scenarios
=========================================================
Real email scenarios across three difficulty tiers:
  easy   → single inbox, clear priority signals
  medium → mixed inbox, ambiguous signals, multiple action types
  hard   → large noisy inbox, conflicting signals, multi-step triage
"""

from typing import Dict, List, Any

# ──────────────────────────────────────────────
# EMAIL CORPUS
# ──────────────────────────────────────────────

EMAIL_CORPUS: List[Dict[str, Any]] = [
    # --- URGENT / HIGH PRIORITY ---
    {
        "id": "e001",
        "from": "cto@enterprise.com",
        "to": "support@company.com",
        "subject": "URGENT: Production database is down — revenue impact",
        "body": (
            "Our entire production database cluster went offline 20 minutes ago. "
            "We are losing $50,000 per minute. This is a P0 incident. "
            "We need your on-call engineer on a call immediately. "
            "Contact: +1-555-0100. This is escalated to your CEO."
        ),
        "timestamp": "2024-03-15T09:02:00Z",
        "has_attachment": False,
        "labels": [],
        "true_priority": "urgent",
        "true_category": "incident",
        "true_action": "escalate",
        "requires_response": True,
    },
    {
        "id": "e002",
        "from": "legal@bigcorp.com",
        "to": "contracts@company.com",
        "subject": "Contract deadline TODAY 5pm — signature required",
        "body": (
            "Per our agreement, the MSA must be countersigned by 5pm today or "
            "the deal is void. Please DocuSign immediately. "
            "This is a $2M contract. I have been trying to reach your team for 3 days."
        ),
        "timestamp": "2024-03-15T08:45:00Z",
        "has_attachment": True,
        "labels": [],
        "true_priority": "urgent",
        "true_category": "legal",
        "true_action": "escalate",
        "requires_response": True,
    },
    {
        "id": "e003",
        "from": "security-alerts@company.com",
        "to": "security@company.com",
        "subject": "Alert: Suspicious login attempt — root account",
        "body": (
            "We detected a login attempt from IP 45.33.32.156 (Russia) "
            "for the root account. MFA was bypassed using a session token. "
            "User account may be compromised. Immediate review required."
        ),
        "timestamp": "2024-03-15T09:15:00Z",
        "has_attachment": False,
        "labels": ["auto-generated"],
        "true_priority": "urgent",
        "true_category": "security",
        "true_action": "escalate",
        "requires_response": True,
    },
    # --- HIGH PRIORITY ---
    {
        "id": "e004",
        "from": "vp.sales@enterprise.com",
        "to": "sales@company.com",
        "subject": "Follow-up: Enterprise pilot renewal next week",
        "body": (
            "Hi team, our 90-day pilot expires next Friday. "
            "We need the renewal quote and SLA document before our board meeting Thursday. "
            "We're evaluating two other vendors as well."
        ),
        "timestamp": "2024-03-15T08:00:00Z",
        "has_attachment": False,
        "labels": [],
        "true_priority": "high",
        "true_category": "sales",
        "true_action": "reply",
        "requires_response": True,
    },
    {
        "id": "e005",
        "from": "hr@company.com",
        "to": "allstaff@company.com",
        "subject": "Benefits enrollment closes Friday — action required",
        "body": (
            "This is a reminder that open enrollment for health benefits closes this Friday. "
            "If you do not select a plan you will be auto-enrolled in the Basic plan. "
            "Log in to HR portal to make your selection."
        ),
        "timestamp": "2024-03-13T10:00:00Z",
        "has_attachment": False,
        "labels": ["hr", "bulk"],
        "true_priority": "high",
        "true_category": "hr",
        "true_action": "archive",
        "requires_response": False,
    },
    {
        "id": "e006",
        "from": "customer_jane@startup.io",
        "to": "support@company.com",
        "subject": "Bug: Data export broken since yesterday's update",
        "body": (
            "Since the update yesterday, clicking Export CSV gives a 500 error. "
            "I have 10,000 records I need for an investor presentation tomorrow. "
            "Please fix this ASAP or provide an alternative way to export."
        ),
        "timestamp": "2024-03-15T07:30:00Z",
        "has_attachment": False,
        "labels": [],
        "true_priority": "high",
        "true_category": "bug",
        "true_action": "reply",
        "requires_response": True,
    },
    # --- MEDIUM PRIORITY ---
    {
        "id": "e007",
        "from": "partner@vendor.com",
        "to": "partnerships@company.com",
        "subject": "Partnership proposal — integration opportunity",
        "body": (
            "We'd love to explore a technical integration between our platforms. "
            "We have 500K mutual customers. Could we schedule a call next week?"
        ),
        "timestamp": "2024-03-14T14:00:00Z",
        "has_attachment": True,
        "labels": [],
        "true_priority": "medium",
        "true_category": "business",
        "true_action": "reply",
        "requires_response": True,
    },
    {
        "id": "e008",
        "from": "accounting@company.com",
        "to": "team-leads@company.com",
        "subject": "Q1 expense reports due March 31",
        "body": (
            "Please remind your teams that Q1 expense reports must be submitted by March 31. "
            "Use the new Concur portal. Late submissions will not be reimbursed."
        ),
        "timestamp": "2024-03-12T09:00:00Z",
        "has_attachment": False,
        "labels": ["internal", "bulk"],
        "true_priority": "medium",
        "true_category": "admin",
        "true_action": "archive",
        "requires_response": False,
    },
    {
        "id": "e009",
        "from": "press@journalist.com",
        "to": "pr@company.com",
        "subject": "Interview request for TechCrunch article on AI startups",
        "body": (
            "I'm writing a piece on emerging AI startups for TechCrunch. "
            "Would your CEO be available for a 20-minute phone interview this week? "
            "The piece publishes April 1."
        ),
        "timestamp": "2024-03-14T11:00:00Z",
        "has_attachment": False,
        "labels": [],
        "true_priority": "medium",
        "true_category": "pr",
        "true_action": "reply",
        "requires_response": True,
    },
    # --- LOW PRIORITY ---
    {
        "id": "e010",
        "from": "newsletter@saas-news.com",
        "to": "user@company.com",
        "subject": "SaaS Weekly: Top 10 tools you need in 2024",
        "body": "Check out this week's top SaaS tools. Unsubscribe here.",
        "timestamp": "2024-03-15T06:00:00Z",
        "has_attachment": False,
        "labels": ["newsletter"],
        "true_priority": "low",
        "true_category": "newsletter",
        "true_action": "delete",
        "requires_response": False,
    },
    {
        "id": "e011",
        "from": "no-reply@linkedin.com",
        "to": "user@company.com",
        "subject": "You have 5 new connection requests",
        "body": "People you may know want to connect with you on LinkedIn.",
        "timestamp": "2024-03-15T05:00:00Z",
        "has_attachment": False,
        "labels": ["social"],
        "true_priority": "low",
        "true_category": "social",
        "true_action": "delete",
        "requires_response": False,
    },
    {
        "id": "e012",
        "from": "notifications@github.com",
        "to": "dev@company.com",
        "subject": "[org/repo] PR #1234 approved by reviewer",
        "body": "Your pull request was approved. Merge when ready.",
        "timestamp": "2024-03-15T08:10:00Z",
        "has_attachment": False,
        "labels": ["github", "automated"],
        "true_priority": "low",
        "true_category": "notification",
        "true_action": "archive",
        "requires_response": False,
    },
    {
        "id": "e013",
        "from": "accounts@aws.amazon.com",
        "to": "billing@company.com",
        "subject": "Your AWS bill for February: $12,450.22",
        "body": (
            "Your AWS invoice for February is ready. Amount due: $12,450.22. "
            "Payment will be charged to your card on file on March 15."
        ),
        "timestamp": "2024-03-10T00:00:00Z",
        "has_attachment": True,
        "labels": ["billing", "automated"],
        "true_priority": "medium",
        "true_category": "billing",
        "true_action": "archive",
        "requires_response": False,
    },
    {
        "id": "e014",
        "from": "customer_bob@midsize.com",
        "to": "support@company.com",
        "subject": "How do I export data to Excel?",
        "body": (
            "Hi, I'm trying to get my data into Excel format. "
            "I see CSV export but I prefer xlsx. Is that possible?"
        ),
        "timestamp": "2024-03-14T16:00:00Z",
        "has_attachment": False,
        "labels": [],
        "true_priority": "low",
        "true_category": "support",
        "true_action": "reply",
        "requires_response": True,
    },
    {
        "id": "e015",
        "from": "ceo@enterprise.com",
        "to": "sales@company.com",
        "subject": "Vendor comparison — need pricing by EOD",
        "body": (
            "We are making a final vendor decision today. "
            "Please send me your enterprise pricing tier and any available discounts. "
            "We are comparing you against Competitor X and Y."
        ),
        "timestamp": "2024-03-15T09:00:00Z",
        "has_attachment": False,
        "labels": [],
        "true_priority": "urgent",
        "true_category": "sales",
        "true_action": "escalate",
        "requires_response": True,
    },
]

# ──────────────────────────────────────────────
# TASK DEFINITIONS
# ──────────────────────────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {
    # ── EASY ──────────────────────────────────
    "easy_triage": {
        "name": "easy_triage",
        "description": (
            "Triage a small inbox of 5 clearly labeled emails. "
            "Assign the correct priority (urgent/high/medium/low) to each email. "
            "Max 10 steps."
        ),
        "difficulty": "easy",
        "max_steps": 10,
        "email_ids": ["e001", "e010", "e004", "e011", "e014"],
        "required_actions": ["label"],  # only labeling required
        "grader_config": {
            "priority_weight": 1.0,
            "action_weight": 0.0,
            "response_quality_weight": 0.0,
        },
    },
    # ── MEDIUM ────────────────────────────────
    "medium_triage": {
        "name": "medium_triage",
        "description": (
            "Triage 8 emails: assign priority AND choose the correct action "
            "(reply/archive/delete/escalate) for each. Max 16 steps."
        ),
        "difficulty": "medium",
        "max_steps": 16,
        "email_ids": ["e001", "e006", "e007", "e008", "e010", "e012", "e013", "e015"],
        "required_actions": ["label", "take_action"],
        "grader_config": {
            "priority_weight": 0.5,
            "action_weight": 0.5,
            "response_quality_weight": 0.0,
        },
    },
    # ── HARD ──────────────────────────────────
    "hard_triage": {
        "name": "hard_triage",
        "description": (
            "Full triage of 12 emails: assign priority, choose action, AND "
            "draft a reply body for all emails that require one. "
            "Reply quality is scored for relevance and professionalism. Max 30 steps."
        ),
        "difficulty": "hard",
        "max_steps": 30,
        "email_ids": [
            "e001", "e002", "e003", "e004", "e006",
            "e007", "e009", "e010", "e012", "e013", "e014", "e015",
        ],
        "required_actions": ["label", "take_action", "reply"],
        "grader_config": {
            "priority_weight": 0.35,
            "action_weight": 0.35,
            "response_quality_weight": 0.30,
        },
    },
}

PRIORITY_LEVELS = ["urgent", "high", "medium", "low"]
ACTION_TYPES = ["reply", "archive", "delete", "escalate", "label", "skip"]