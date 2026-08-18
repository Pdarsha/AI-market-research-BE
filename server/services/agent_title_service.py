import re
from datetime import datetime
from typing import Any, Dict, List, Optional


# Common words to trim/clean from agent names to form a clean "Purpose"
_AGENT_SUFFIX_PATTERNS = [
    r"\b(agent|workflow|bot|assistant|pipeline|service|tool|system)\b",
]

# Common verbs and prefixes to strip from instructions/queries when extracting a concise "Task"
_PROMPT_PREFIX_PATTERNS = [
    r"^(please\s+)?(analyze|analyse|evaluate|review|summarize|summarise|extract|compare|calculate|compute|generate|track|monitor|perform|provide|check|create|build|run|execute|inspect|identify|find)\s+(the\s+|a\s+|an\s+|our\s+|all\s+|key\s+|comprehensive\s+|detailed\s+)?",
    r"^(executive\s+summary\s+of|summary\s+of|overview\s+of|report\s+on|breakdown\s+of|analysis\s+of|deep\s+dive\s+into|review\s+of)\s+",
    r"^(execute\s+agent\s+workflow|run\s+agent\s+workflow|run\s+agent|execute\s+agent):\s*",
]

# Stop phrases that are unhelpful in titles
_UNHELPFUL_TASK_PHRASES = {
    "agent workflow",
    "agent run",
    "run agent",
    "execute workflow",
    "undefined",
    "null",
    "none",
}


def _clean_purpose(agent_name: Optional[str]) -> str:
    """Extract a clean, readable Purpose from an agent name."""
    if not agent_name:
        return "Agent Task"

    cleaned = agent_name.strip()

    # If it starts with "Agent ·", strip that out first to avoid duplication
    if cleaned.startswith("Agent ·"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("Agent:"):
        cleaned = cleaned[6:].strip()
    elif cleaned.startswith("Agent -"):
        cleaned = cleaned[7:].strip()

    # Strip command-like prefixes
    for pattern in _PROMPT_PREFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # Clean suffixes like "Agent", "Workflow", "Bot", "Assistant"
    # But only if stripping them leaves at least one word
    words = cleaned.split()
    if len(words) > 1:
        last_word = words[-1].lower().rstrip("s")
        if last_word in ("agent", "workflow", "bot", "assistant", "pipeline", "service", "system"):
            cleaned = " ".join(words[:-1]).strip()

    # Common substitutions for elegance
    normalized_lower = cleaned.lower()
    if normalized_lower in ("financial analyst", "financial analysis agent"):
        return "Financial Analysis"
    if normalized_lower in ("market researcher", "market research agent"):
        return "Market Research"
    if normalized_lower in ("contract reviewer", "contract review agent"):
        return "Contract Review"
    if normalized_lower in ("document reviewer", "document review agent"):
        return "Document Review"

    # Title-case clean purpose
    return cleaned.strip() or "Agent Task"


def _extract_task_from_text(text: Optional[str], purpose: str) -> Optional[str]:
    """Extract a concise 2-4 word task/topic from instruction or query text."""
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # 1. Strip prompt prefix verbs
    for pattern in _PROMPT_PREFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # 2. Take only the first sentence or first clause before punctuation / connector
    # e.g., "Analyze technology market trends and identify major competitors." -> "technology market trends"
    first_clause = re.split(r"[.\n;]|(?:\s+and\s+(?:identify|evaluate|compare|calculate|provide|summarize))", cleaned, flags=re.IGNORECASE)[0].strip()

    # 3. Clean trailing punctuation
    first_clause = re.sub(r"[\.,;:!?]+$", "", first_clause).strip()

    if not first_clause or first_clause.lower() in _UNHELPFUL_TASK_PHRASES:
        return None

    # 4. Remove redundant purpose words from the task to prevent "Morning Brief — Morning Brief"
    purpose_tokens = set(purpose.lower().split())
    task_words = first_clause.split()

    filtered_words = [w for w in task_words if w.lower() not in purpose_tokens]
    if filtered_words and len(filtered_words) >= 1:
        task_candidate = " ".join(filtered_words)
    else:
        task_candidate = first_clause

    # 5. Trim to ~3-5 words / max 35 characters
    words = task_candidate.split()
    if len(words) > 5:
        task_candidate = " ".join(words[:4])

    if len(task_candidate) > 35:
        task_candidate = task_candidate[:32].rstrip() + "…"

    # Title-case the task
    task_title = " ".join(
        w.capitalize() if w.lower() not in ("and", "of", "the", "in", "on", "for", "to", "with", "vs", "a", "an") else w.lower()
        for w in task_candidate.split()
    )
    # Ensure first character is uppercase
    if task_title:
        task_title = task_title[0].upper() + task_title[1:]

    # Check if task is essentially identical to purpose
    if task_title.lower() == purpose.lower():
        return None

    return task_title if len(task_title) >= 3 else None


def _format_date(dt: Optional[datetime]) -> str:
    """Format execution date concisely as 'Aug 18'."""
    d = dt or datetime.utcnow()
    return d.strftime("%b %d")


def generate_agent_chat_title(
    agent_name: Optional[str] = None,
    instructions: Optional[str] = None,
    query: Optional[str] = None,
    document_names: Optional[List[str]] = None,
    trigger_source: Optional[str] = None,
    execution_time: Optional[datetime] = None,
    max_length: int = 75,
) -> str:
    """Generate a clean, deterministic, human-readable Agent chat title.

    Format:
        Agent · [Purpose] — [Task / Context]

    Examples:
        - Agent · Market Research — Technology Trends
        - Agent · Financial Analysis — Quarterly Performance
        - Agent · Contract Review — Risk Analysis
        - Agent · Morning Brief — Aug 18
        - Agent · Document Review — Key Findings

    Guarantees:
        - "Agent · " prefix is ALWAYS present.
        - Never uses full raw prompt text.
        - Never contains UUIDs, document IDs, or internal endpoints.
        - Never duplicates words between Purpose and Task.
        - Keeps title length under max_length (default 75 chars).
    """
    purpose = _clean_purpose(agent_name)

    # 1. Try to extract concise task from query or instructions
    task = None
    # If a specific user query was provided and differs from instructions, use it first
    if query and query.strip() and query.strip().lower() != (instructions or "").strip().lower():
        task = _extract_task_from_text(query, purpose)

    if not task and instructions and instructions.strip():
        task = _extract_task_from_text(instructions, purpose)

    # 2. If no task extracted, check document names if available
    if not task and document_names:
        # Check if multiple reports share a clean theme
        cleaned_doc_names = []
        for name in document_names[:3]:
            # Remove file extension
            base = re.sub(r"\.[a-zA-Z0-9]+$", "", str(name)).replace("_", " ").replace("-", " ")
            cleaned_doc_names.append(base.strip())

        if any("quarter" in n.lower() or "q1" in n.lower() or "q2" in n.lower() or "q3" in n.lower() or "q4" in n.lower() for n in cleaned_doc_names):
            task = "Quarterly Reports"
        elif any("financial" in n.lower() or "statement" in n.lower() for n in cleaned_doc_names):
            task = "Financial Statements"
        elif any("contract" in n.lower() or "agreement" in n.lower() for n in cleaned_doc_names):
            task = "Agreements"
        elif cleaned_doc_names:
            first_doc = cleaned_doc_names[0]
            words = first_doc.split()
            if len(words) <= 3:
                task = first_doc.title()

    # 3. If trigger source is scheduled or periodic, append date if task is generic
    if trigger_source in ("scheduled", "periodic", "cron", "background") or not task:
        date_str = _format_date(execution_time)
        if not task:
            task = date_str
        elif trigger_source == "scheduled":
            # e.g., "Agent · Daily Market Brief — Aug 18"
            if any(term in purpose.lower() for term in ("daily", "weekly", "morning", "brief", "digest")):
                task = date_str

    # 4. Construct final title
    if task:
        full_title = f"Agent · {purpose} — {task}"
    else:
        full_title = f"Agent · {purpose}"

    # 5. Length constraint
    if len(full_title) > max_length:
        full_title = full_title[: max_length - 1].rstrip() + "…"

    return full_title
