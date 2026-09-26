"""
Tests for the /api/execute endpoint and tier-to-model mapping.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from router.config import MODEL_PROFILES, TIER_DEFINITIONS


class TestTierToModelMapping:
    """Verify the tier→model mapping is consistent."""

    def test_all_local_tiers_have_model_profiles(self):
        for tier_name, td in TIER_DEFINITIONS.items():
            if td.is_local:
                assert td.preferred_model_profiles, (
                    f"Local tier {tier_name} missing preferred_model_profiles"
                )

    def test_all_mapped_profiles_exist(self):
        for tier_name, td in TIER_DEFINITIONS.items():
            for profile_name in td.preferred_model_profiles:
                assert profile_name in MODEL_PROFILES, (
                    f"Tier {tier_name} references unknown profile {profile_name!r}"
                )

    def test_local_coding_uses_coder_model(self):
        td = TIER_DEFINITIONS["local_coding"]
        profile_name = td.preferred_model_profiles[0]
        profile = MODEL_PROFILES[profile_name]
        assert "coding" in profile.capabilities

    def test_local_fast_uses_fast_classifier(self):
        td = TIER_DEFINITIONS["local_fast"]
        assert td.preferred_model_profiles[0] == "rlcd_fast"

    def test_splash_local_profile_is_available_for_explicit_selection(self):
        profile = MODEL_PROFILES["splash_local"]
        assert profile.model_id == "incoai/Qwen3.8-27B-Splash"
        assert profile.provider == "openai"
        assert profile.role == "executor"
        assert profile.billing == "local"
        assert {"general", "coding", "reasoning", "tools"}.issubset(
            profile.capabilities
        )

    def test_all_tiers_have_system_prompts(self):
        from server.app import TIER_SYSTEM_PROMPTS
        for tier_name in TIER_DEFINITIONS:
            if TIER_DEFINITIONS[tier_name].is_local:
                assert tier_name in TIER_SYSTEM_PROMPTS, (
                    f"Tier {tier_name} missing system prompt"
                )


class TestEscalationIntegration:
    """Test that escalation chains produce valid model transitions."""

    def test_local_coding_escalation_targets_exist(self):
        from router.decision import DEFAULT_ESCALATION
        from router.config import TIER_DEFINITIONS

        esc = DEFAULT_ESCALATION["local_coding"]
        next_tier = esc["next_tier"]
        assert next_tier in TIER_DEFINITIONS
        assert next_tier == "cloud_standard"

    def test_full_escalation_chain_all_valid(self):
        from router.decision import DEFAULT_ESCALATION
        from router.config import TIER_DEFINITIONS

        tier = "local_fast"
        visited = [tier]
        while tier in DEFAULT_ESCALATION:
            next_tier = DEFAULT_ESCALATION[tier].get("next_tier")
            if next_tier is None:
                break
            assert next_tier in TIER_DEFINITIONS, (
                f"Escalation from {tier} to {next_tier} references unknown tier"
            )
            assert next_tier not in visited, (
                f"Escalation cycle detected: {visited} → {next_tier}"
            )
            visited.append(next_tier)
            tier = next_tier

        assert len(visited) >= 5  # at least 5 tiers in the chain


# ===================================================================
# Task 10: deprecation of router-owned execution endpoints
#
# /api/execute and /api/route-and-execute stay functional legacy
# endpoints, but they must explicitly signal deprecation in their
# responses and OpenAPI docs.  Cloud execution belongs to the host
# (PI); the router only ever executed local tiers.
# ===================================================================

_MOCK_TEXT_RESULT = {
    "mode": "freeform_autoregressive",
    "elapsed_ms": 10.0,
    "total_tokens": 4,
    "tokens_per_second": 400.0,
    "text": "mock legacy execution output",
}

_MOCK_CLASSIFY_RESULT = {
    "parsed_json": {
        "task_type": {"value": "code_edit", "prob": 0.95},
        "complexity": {"value": "low", "prob": 0.92},
        "coding": {"value": "substantial", "prob": 0.93},
        "reasoning_depth": {"value": "direct", "prob": 0.91},
        "fresh_information": {"value": "none", "prob": 0.97},
        "repo_access": {"value": "modify", "prob": 0.94},
        "tool_use": {"value": "shell", "prob": 0.90},
        "privacy": {"value": "ordinary", "prob": 0.98},
        "latency": {"value": "normal", "prob": 0.89},
        "failure_cost": {"value": "low", "prob": 0.92},
        "output_contract": {"value": "exact_patch", "prob": 0.93},
    },
    "elapsed_ms": 50.0,
    "is_valid_json": True,
    "schema_match": True,
}


def _patch_legacy_engine():
    """Mock the lazily imported MLX engine so the legacy execution
    endpoints run without loading any model."""
    import core.engine as engine

    saved = {}
    for name in (
        "MODEL_ID", "run_text_generation", "run_parallel_for_model",
        "run_parallel_generation", "list_loaded_models",
    ):
        saved[name] = getattr(engine, name)

    def mock_run_text(model_id, prompt, max_tokens=2048, temperature=0.2,
                      system_prompt=None):
        return dict(_MOCK_TEXT_RESULT)

    def mock_run_parallel_for(model_id, context, schema, temperature=1.0):
        return dict(_MOCK_CLASSIFY_RESULT)

    engine.MODEL_ID = "mock://legacy/default-model"
    engine.run_text_generation = mock_run_text
    engine.run_parallel_for_model = mock_run_parallel_for
    engine.run_parallel_generation = (
        lambda context, schema, temperature=1.0: dict(_MOCK_CLASSIFY_RESULT)
    )
    engine.list_loaded_models = lambda: []
    return saved


def _restore_legacy_engine(saved):
    import core.engine as engine
    for name, value in saved.items():
        setattr(engine, name, value)


class TestExecutionEndpointDeprecation:
    """Legacy execution endpoints: functional, but explicitly deprecated."""

    def setup_method(self):
        self._saved = _patch_legacy_engine()
        import server.app as app_module
        self.app_module = app_module
        app_module._router_classifier = None
        self.client = TestClient(app_module.app)

    def teardown_method(self):
        self.app_module._router_classifier = None
        _restore_legacy_engine(self._saved)

    def test_legacy_execution_endpoints_marked_deprecated_in_docs(self):
        spec = self.client.get("/openapi.json").json()
        assert spec["paths"]["/api/execute"]["post"]["deprecated"] is True
        assert (
            spec["paths"]["/api/route-and-execute"]["post"]["deprecated"]
            is True
        )
        # Descriptions must explain the deprecation and the migration path.
        for path in ("/api/execute", "/api/route-and-execute"):
            description = (
                spec["paths"][path]["post"].get("description") or ""
            ).lower()
            assert "deprecated" in description, path
            assert "/api/route" in description, path
        # The production route is not deprecated.
        assert spec["paths"]["/api/route"]["post"].get("deprecated") is not True

    def test_execute_response_signals_deprecation(self):
        resp = self.client.post("/api/execute", json={
            "prompt": "Fix the login handler",
            "tier": "local_coding",
        })
        assert resp.status_code == 200
        # RFC 9745 Deprecation header (versionless form).
        assert resp.headers["Deprecation"] == "true"
        # RFC 8594 Sunset header: provisional removal horizon.
        assert "Sunset" in resp.headers
        body = resp.json()
        assert body["deprecated"] is True
        assert "deprecation" in body
        # The endpoint remains functional.
        assert body["text"] == _MOCK_TEXT_RESULT["text"]

    def test_route_and_execute_response_signals_deprecation(self):
        resp = self.client.post("/api/route-and-execute", json={
            "prompt": "Fix the login bug in auth.py",
        })
        assert resp.status_code == 200
        assert resp.headers["Deprecation"] == "true"
        assert "Sunset" in resp.headers
        body = resp.json()
        assert body["deprecated"] is True
        assert "deprecation" in body
        assert body["execution"] is not None

    def test_route_and_execute_override_path_signals_deprecation(self):
        resp = self.client.post("/api/route-and-execute", json={
            "prompt": "Summarize this output for me",
            "model_override": "local_coding",
        })
        assert resp.status_code == 200
        assert resp.headers["Deprecation"] == "true"
        body = resp.json()
        assert body["deprecated"] is True
        assert "deprecation" in body

    def test_sunset_header_is_configurable(self):
        import os
        saved = os.environ.get("ROUTER_LEGACY_SUNSET")
        os.environ["ROUTER_LEGACY_SUNSET"] = "Wed, 01 Jan 2025 00:00:00 GMT"
        try:
            resp = self.client.post("/api/execute", json={
                "prompt": "hello", "tier": "local_fast",
            })
        finally:
            if saved is None:
                os.environ.pop("ROUTER_LEGACY_SUNSET", None)
            else:
                os.environ["ROUTER_LEGACY_SUNSET"] = saved
        assert resp.status_code == 200
        assert resp.headers["Sunset"] == "Wed, 01 Jan 2025 00:00:00 GMT"


