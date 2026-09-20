"""
RLCD schema definition for the routing classifier.

Each field is a constrained enum or boolean.  The RLCD engine
broadcasts a single KV-cache prefix across all fields and scores
only the allowed candidates, so this schema directly controls
both accuracy and latency.

Design principles:
  - Every enum value is a short lowercase_snake token to keep
    candidate sets small and tokenizer-friendly.
  - Fields are independent (no cross-field constraints) so the
    parallel broadcast works cleanly.
  - ~11 fields keeps latency under 200 ms on the 1.5 B model
    while still capturing the decision surface.
"""

from typing import Dict, Any


# ---------------------------------------------------------------------------
# The canonical router schema consumed by StructuredSchema / RLCD
# ---------------------------------------------------------------------------

ROUTER_SCHEMA: Dict[str, Dict[str, Any]] = {
    "task_type": {
        "type": "enum",
        "choices": [
            "code_edit",
            "code_review",
            "debugging",
            "explanation",
            "planning",
            "research",
            "extraction",
            "classification",
            "creative",
            "operations",
            "unknown",
        ],
        "description": "What the user wants done. code_edit=write or fix code, debugging=find and fix a bug, explanation=answer a question or describe how something works, planning=design or architecture, research=look up current facts, extraction=pull data from text, classification=categorize items",
    },
    "complexity": {
        "type": "enum",
        "choices": ["trivial", "low", "medium", "high", "critical"],
        "description": "How hard the task is. trivial=one-line change or simple question, low=small function or straightforward fix, medium=multi-file change or non-obvious bug, high=large refactor or architecture decision, critical=production incident or security fix",
    },
    "coding": {
        "type": "enum",
        "choices": ["none", "light", "substantial", "dominant"],
        "description": "Whether the task involves writing or modifying code. none=question or research with no code, light=small config or one-line script, substantial=writing a function or fixing a bug, dominant=writing a module or refactoring files",
    },
    "reasoning_depth": {
        "type": "enum",
        "choices": ["direct", "multi_step", "deep", "adversarial"],
        "description": "How much thinking is needed. direct=simple lookup or single-step answer, multi_step=a few logical steps or a small algorithm, deep=complex analysis with trade-offs or cross-system reasoning, adversarial=security analysis or finding subtle bugs",
    },
    "fresh_information": {
        "type": "enum",
        "choices": ["none", "useful", "required"],
        "description": "Whether current or recent web data is needed. none=the task uses only provided context, useful=web search might help but is not essential, required=the task cannot be done without current web data such as recent news or latest API docs",
    },
    "repo_access": {
        "type": "enum",
        "choices": ["none", "inspect", "modify"],
        "description": "Whether the task needs repository files. none=standalone question or task, inspect=read files to understand code, modify=edit or create files in the repository",
    },
    "tool_use": {
        "type": "enum",
        "choices": ["none", "read_only", "shell", "browser", "multiple"],
        "description": "What tools the task will use. none=pure text response, read_only=read files only, shell=run commands like tests or builds, browser=search the web, multiple=combine shell and browser or other tools",
    },
    "privacy": {
        "type": "enum",
        "choices": ["public", "ordinary", "sensitive", "local_only"],
        "description": "How sensitive the data is. public=public code or general knowledge, ordinary=typical work data with no restrictions, sensitive=credentials or personal data, local_only=must not leave the local machine",
    },
    "latency": {
        "type": "enum",
        "choices": ["interactive", "normal", "asynchronous"],
        "description": "How fast the user needs a response. interactive=user is actively waiting and needs a quick reply, normal=standard response time is fine, asynchronous=background or batch task where speed does not matter",
    },
    "failure_cost": {
        "type": "enum",
        "choices": ["low", "medium", "high", "very_high"],
        "description": "What happens if the answer is wrong. low=the user can easily check and retry, medium=minor wasted effort or delay, high=broken production or incorrect deployment, very_high=data loss or security breach",
    },
    "output_contract": {
        "type": "enum",
        "choices": ["freeform", "structured", "exact_patch", "verified_artifact"],
        "description": "What format the output must be in. freeform=natural language or explanation, structured=JSON or specific data format, exact_patch=a code diff that must apply cleanly, verified_artifact=output that must pass tests or validation",
    },
}


# ---------------------------------------------------------------------------
# Slim fast schema – the 4 fields the 1.5B model handles reliably.
# Used by stage 1 (fast classifier).  Fewer fields = higher
# confidence and faster inference (~100–150 ms vs ~330 ms for 11).
# ---------------------------------------------------------------------------

FAST_ROUTER_SCHEMA: Dict[str, Dict[str, Any]] = {
    "coding": ROUTER_SCHEMA["coding"],
    "fresh_information": ROUTER_SCHEMA["fresh_information"],
    "repo_access": ROUTER_SCHEMA["repo_access"],
    "privacy": ROUTER_SCHEMA["privacy"],
}

# Defaults for the 7 fields the fast schema omits.
# When fast classification is accepted, these fill the gaps.
FAST_SCHEMA_DEFAULTS: Dict[str, str] = {
    "task_type": "unknown",
    "complexity": "medium",
    "reasoning_depth": "direct",
    "tool_use": "none",
    "latency": "normal",
    "failure_cost": "low",
    "output_contract": "freeform",
}


# ---------------------------------------------------------------------------
# Ordered field names (deterministic iteration for RLCD broadcast)
# ---------------------------------------------------------------------------

ROUTER_FIELD_ORDER: list[str] = list(ROUTER_SCHEMA.keys())
FAST_ROUTER_FIELD_ORDER: list[str] = list(FAST_ROUTER_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Critical fields – low confidence on these triggers smart-router escalation
# ---------------------------------------------------------------------------

CRITICAL_FIELDS: frozenset[str] = frozenset({
    "privacy",
    "fresh_information",
    "repo_access",
    "failure_cost",
    "output_contract",
})

# Critical fields within the fast schema
FAST_CRITICAL_FIELDS: frozenset[str] = frozenset({
    "privacy",
    "fresh_information",
    "repo_access",
})


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_router_output(raw: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Validate that every field in *raw* is an allowed enum value.

    Returns ``(validated, errors)``.  ``validated`` maps field name → value
    for every field that passed; ``errors`` lists the problems.
    """
    validated: dict[str, str] = {}
    errors: list[str] = []

    for field_name, spec in ROUTER_SCHEMA.items():
        value = raw.get(field_name)
        if value is None:
            errors.append(f"missing field: {field_name}")
            continue

        choices = spec["choices"]
        str_value = str(value).strip()
        if str_value not in choices:
            errors.append(
                f"{field_name}={str_value!r} not in {choices}"
            )
            continue
        validated[field_name] = str_value

    extra = set(raw.keys()) - set(ROUTER_SCHEMA.keys()) - {"confidence", "abstain"}
    if extra:
        errors.append(f"unexpected fields: {sorted(extra)}")

    return validated, errors


def build_router_preset(
    context: str,
    title: str = "PI Router Classification",
) -> Dict[str, Any]:
    """Build a preset dict compatible with the existing RLCD benchmark
    runner, useful for offline evaluation of the router schema."""
    return {
        "id": "pi_router",
        "title": title,
        "description": "Multi-model routing classification for PI harness",
        "context": context,
        "schema": ROUTER_SCHEMA,
    }
