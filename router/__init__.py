"""
Multi-model routing layer for PI (coding/agent harness).

Sits in front of the RLCD engine to classify task requirements,
then maps capabilities to a deterministic model tier via a pure
policy function.  The RLCD model predicts *what the task needs*;
the policy decides *which tier can satisfy those needs*.
"""

from router.router_schema import ROUTER_SCHEMA, FAST_ROUTER_SCHEMA, FAST_SCHEMA_DEFAULTS
from router.signals import RuntimeSignals
from router.decision import RouteDecision, TierRequirement
from router.config import (
    ModelProfile,
    MODEL_PROFILES,
    ConfidenceThresholds,
    CONFIDENCE_THRESHOLDS,
    TierDefinition,
    TIER_DEFINITIONS,
)
from router.policy import choose_route
from router.escalation import ExecutionState, should_escalate, escalate
from router.classifier import RouterClassifier, ClassificationResult
from router.telemetry import RoutingEvent, TelemetryLog

__all__ = [
    "ROUTER_SCHEMA",
    "RuntimeSignals",
    "RouteDecision",
    "TierRequirement",
    "ModelProfile",
    "MODEL_PROFILES",
    "ConfidenceThresholds",
    "CONFIDENCE_THRESHOLDS",
    "TierDefinition",
    "TIER_DEFINITIONS",
    "choose_route",
    "ExecutionState",
    "should_escalate",
    "escalate",
    "RouterClassifier",
    "ClassificationResult",
    "RoutingEvent",
    "TelemetryLog",
]
