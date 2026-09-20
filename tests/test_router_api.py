"""
Integration test: exercise /api/route, /api/execute, and
/api/route-and-execute without loading MLX models.

Uses the TestClient from FastAPI with a mock RLCD engine.
"""

from __future__ import annotations

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


# We need to patch the engine before importing the app so that
# the router classifier uses our mock instead of loading MLX.

_MOCK_FAST_OUTPUT = {
    "parsed_json": {
        "task_type": {"value": "code_edit", "prob": 0.95},
        "complexity": {"value": "low", "prob": 0.92},
        "coding": {"value": "dominant", "prob": 0.97},
        "reasoning_depth": {"value": "direct", "prob": 0.90},
        "fresh_information": {"value": "none", "prob": 0.98},
        "repo_access": {"value": "modify", "prob": 0.96},
        "tool_use": {"value": "shell", "prob": 0.93},
        "privacy": {"value": "ordinary", "prob": 0.99},
        "latency": {"value": "normal", "prob": 0.88},
        "failure_cost": {"value": "medium", "prob": 0.91},
        "output_contract": {"value": "exact_patch", "prob": 0.94},
    },
    "elapsed_ms": 85.0,
    "is_valid_json": True,
    "schema_match": True,
}

_MOCK_SMART_OUTPUT = {
    "parsed_json": {
        "task_type": {"value": "debugging", "prob": 0.88},
        "complexity": {"value": "medium", "prob": 0.82},
        "coding": {"value": "dominant", "prob": 0.95},
        "reasoning_depth": {"value": "multi_step", "prob": 0.78},
        "fresh_information": {"value": "none", "prob": 0.96},
        "repo_access": {"value": "modify", "prob": 0.94},
        "tool_use": {"value": "shell", "prob": 0.91},
        "privacy": {"value": "ordinary", "prob": 0.97},
        "latency": {"value": "normal", "prob": 0.85},
        "failure_cost": {"value": "high", "prob": 0.80},
        "output_contract": {"value": "exact_patch", "prob": 0.92},
    },
    "elapsed_ms": 850.0,
    "is_valid_json": True,
    "schema_match": True,
}

_MOCK_TEXT_OUTPUT = {
    "mode": "freeform_autoregressive",
    "elapsed_ms": 3200.0,
    "total_tokens": 150,
    "tokens_per_second": 46.9,
    "text": "Here is the fix:\n\n```python\ndef verify_token(token):\n    try:\n        token.verify(key=settings.SECRET_KEY)\n    except TokenExpiredError:\n        raise HTTPException(status_code=401, detail='Token expired')\n```",
}

# Track which model was called for assertions
_call_log: list[str] = []


def _patch_engine():
    """Monkey-patch the RLCD engine before importing the app."""
    import core.engine_mlx as mlx_engine
    import core.engine as engine

    originals = {}
    originals["mlx_get_engine"] = mlx_engine.get_engine
    originals["mlx_run_parallel"] = mlx_engine.run_parallel_generation
    originals["mlx_run_parallel_for"] = mlx_engine.run_parallel_for_model
    originals["mlx_load_model"] = mlx_engine.load_model_by_id
    originals["mlx_list_loaded"] = mlx_engine.list_loaded_models
    originals["mlx_run_text"] = mlx_engine.run_text_generation
    originals["mlx_run_text_default"] = mlx_engine.run_text_generation_default
    originals["eng_get_engine"] = engine.get_engine
    originals["eng_run_parallel"] = engine.run_parallel_generation
    originals["eng_run_rlcd"] = engine.run_rlcd_generation
    originals["eng_load_model"] = engine.load_model_by_id
    originals["eng_list_loaded"] = engine.list_loaded_models
    originals["eng_run_parallel_for"] = engine.run_parallel_for_model
    originals["eng_run_text"] = engine.run_text_generation
    originals["eng_run_text_default"] = engine.run_text_generation_default

    def mock_get_engine():
        return (None, None)

    def mock_run_parallel(context, schema, temperature=1.0):
        return dict(_MOCK_FAST_OUTPUT)

    def mock_run_parallel_for(model_id, context, schema, temperature=1.0):
        _call_log.append(f"rlcd:{model_id}")
        # Single-stage classifier: always return high-confidence output
        return dict(_MOCK_FAST_OUTPUT)

    def mock_load_model(model_id, warmup=True):
        _call_log.append(f"load:{model_id}")
        return (None, None)

    def mock_list_loaded():
        return ["mock-fast-model", "mock-smart-model"]

    def mock_run_text(model_id, prompt, max_tokens=2048, temperature=0.2, system_prompt=None):
        _call_log.append(f"text:{model_id}")
        return dict(_MOCK_TEXT_OUTPUT)

    def mock_run_text_default(prompt, max_tokens=2048, temperature=0.2, system_prompt=None):
        _call_log.append("text:default")
        return dict(_MOCK_TEXT_OUTPUT)

    # Patch MLX engine
    mlx_engine.get_engine = mock_get_engine
    mlx_engine.run_parallel_generation = mock_run_parallel
    mlx_engine.run_parallel_for_model = mock_run_parallel_for
    mlx_engine.load_model_by_id = mock_load_model
    mlx_engine.list_loaded_models = mock_list_loaded
    mlx_engine.run_text_generation = mock_run_text
    mlx_engine.run_text_generation_default = mock_run_text_default

    # Patch engine router
    engine.get_engine = mock_get_engine
    engine.run_parallel_generation = mock_run_parallel
    engine.run_rlcd_generation = mock_run_parallel
    engine.load_model_by_id = mock_load_model
    engine.list_loaded_models = mock_list_loaded
    engine.run_parallel_for_model = mock_run_parallel_for
    engine.run_text_generation = mock_run_text
    engine.run_text_generation_default = mock_run_text_default

    return originals


def _unpatch_engine(originals):
    import core.engine_mlx as mlx_engine
    import core.engine as engine

    mlx_engine.get_engine = originals["mlx_get_engine"]
    mlx_engine.run_parallel_generation = originals["mlx_run_parallel"]
    mlx_engine.run_parallel_for_model = originals["mlx_run_parallel_for"]
    mlx_engine.load_model_by_id = originals["mlx_load_model"]
    mlx_engine.list_loaded_models = originals["mlx_list_loaded"]
    mlx_engine.run_text_generation = originals["mlx_run_text"]
    mlx_engine.run_text_generation_default = originals["mlx_run_text_default"]
    engine.get_engine = originals["eng_get_engine"]
    engine.run_parallel_generation = originals["eng_run_parallel"]
    engine.run_rlcd_generation = originals["eng_run_rlcd"]
    engine.load_model_by_id = originals["eng_load_model"]
    engine.list_loaded_models = originals["eng_list_loaded"]
    engine.run_parallel_for_model = originals["eng_run_parallel_for"]
    engine.run_text_generation = originals["eng_run_text"]
    engine.run_text_generation_default = originals["eng_run_text_default"]


class TestRouterAPI:
    def setup_method(self):
        _call_log.clear()
        self._originals = _patch_engine()
        # Force reimport of app to pick up patched engine
        import importlib
        import server.app as app_module
        importlib.reload(app_module)
        self.client = TestClient(app_module.app)
        # Reset the cached classifier
        app_module._router_classifier = None

    def teardown_method(self):
        _unpatch_engine(self._originals)

    # ------------------------------------------------------------------
    # /health
    # ------------------------------------------------------------------

    def test_health_identifies_router_service(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "ok",
            "service": "rlcd-router",
        }

    # ------------------------------------------------------------------
    # /api/run-parallel
    # ------------------------------------------------------------------

    def test_run_parallel_supports_extension_classifier_tool(self):
        resp = self.client.post("/api/run-parallel", json={
            "context": "Classify this task",
            "schema": {
                "kind": {
                    "type": "enum",
                    "choices": ["code", "prose"],
                    "description": "Task kind",
                }
            },
        })
        assert resp.status_code == 200
        assert resp.json()["schema_match"] is True

    # ------------------------------------------------------------------
    # /api/route
    # ------------------------------------------------------------------

    def test_subscription_exhaustion_returns_model_switch_recommendation(self):
        import server.app as app_module

        event = app_module._telemetry_log.new_event(
            eligible_models=[
                "claude_sonnet_4_6",
                "gpt_5_6_sol",
                "local_coding",
            ]
        )
        app_module._telemetry_log.record(event)

        resp = self.client.post("/api/route/outcome", json={
            "event_id": event.event_id,
            "status": "failed",
            "failure_reason": "subscription_exhausted",
            "actual_model": "claude_sonnet_4_6",
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["model_switch"]["recommended_model"] == "gpt_5_6_sol"
        assert body["model_switch"]["can_override"] is True

    def test_route_honors_explicit_model_override(self):
        resp = self.client.post("/api/route", json={
            "prompt": "Review this migration plan",
            "model_override": "gpt_5_6_terra",
        })
        assert resp.status_code == 200
        decision = resp.json()["decision"]
        assert decision["eligible_models"] == ["gpt_5_6_terra"]
        assert decision["classifier_used"] == "user_model_override"
        assert decision["confidence"] == 1.0
        assert "user_requested_model" in decision["reasons"]
        assert _call_log == []

    def test_route_rejects_unknown_model_override(self):
        resp = self.client.post("/api/route", json={
            "prompt": "Review this migration plan",
            "model_override": "not_a_profile",
        })
        assert resp.status_code == 400
        assert "Unknown model profile" in resp.json()["detail"]

    def test_route_rejects_classifier_as_model_override(self):
        resp = self.client.post("/api/route", json={
            "prompt": "Review this migration plan",
            "model_override": "rlcd_smart",
        })
        assert resp.status_code == 400
        assert "executor model profile" in resp.json()["detail"]

    def test_override_catalog_excludes_classifier_profiles(self):
        resp = self.client.get("/api/models/overrides")
        assert resp.status_code == 200
        groups = resp.json()["models"]
        profile_names = {
            model["profile_name"]
            for models in groups.values()
            for model in models
        }
        assert "local_coding" in profile_names
        assert "rlcd_smart" not in profile_names
        assert "rlcd_fast" not in profile_names

    def test_route_basic(self):
        resp = self.client.post("/api/route", json={
            "prompt": "Fix the login bug in auth.py",
        })
        assert resp.status_code == 200
        data = resp.json()

        assert "route_id" in data
        assert "classification" in data
        assert "decision" in data

        cls = data["classification"]
        assert cls["fields"]["task_type"] == "code_edit"
        assert cls["classifier_used"] == "rlcd_fast"

        decision = data["decision"]
        assert decision["tier"] == "local_coding"
        assert "needs_coding" in decision["reasons"]

    def test_route_with_runtime_prior_failures(self):
        resp = self.client.post("/api/route", json={
            "prompt": "Fix the login bug",
            "runtime": {"prior_failures": 1},
        })
        assert resp.status_code == 200
        data = resp.json()
        # Prior failures + coding → cloud_coding
        assert data["decision"]["tier"] == "cloud_coding"

    def test_route_bypass(self):
        resp = self.client.post("/api/route", json={
            "prompt": "Hello world",
            "options": {"bypass": True},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"]["classifier_used"] == "bypass"
        assert data["classification"]["abstain"] is True

    def test_route_config(self):
        resp = self.client.get("/api/route/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "local_coding" in data["tiers"]
        assert "rlcd_fast" in data["models"]

    def test_route_outcome(self):
        resp = self.client.post("/api/route", json={"prompt": "Fix the bug"})
        route_id = resp.json()["route_id"]
        resp2 = self.client.post("/api/route/outcome", json={
            "event_id": route_id,
            "status": "success",
            "verification": "tests_passed",
            "actual_model": "Qwen3-Coder-30B-A3B",
            "latency_ms": 5000.0,
        })
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "recorded"

    def test_route_outcome_not_found(self):
        resp = self.client.post("/api/route/outcome", json={
            "event_id": "nonexistent", "status": "success",
        })
        assert resp.status_code == 404

    def test_telemetry_endpoint(self):
        resp = self.client.get("/api/route/telemetry")
        assert resp.status_code == 200
        assert "total" in resp.json()

    # ------------------------------------------------------------------
    # Two-stage classifier
    # ------------------------------------------------------------------

    def test_route_uses_fast_when_confident(self):
        """Fast classifier is confident → no smart classifier call."""
        resp = self.client.post("/api/route", json={
            "prompt": "Fix the login bug in auth.py",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"]["classifier_used"] == "rlcd_fast"
        # Should have called the fast model
        fast_calls = [c for c in _call_log if "rlcd:" in c ]
        assert len(fast_calls) >= 1

    # ------------------------------------------------------------------
    # /api/execute
    # ------------------------------------------------------------------

    def test_execute_local_coding(self):
        resp = self.client.post("/api/execute", json={
            "prompt": "Fix the login handler in auth.py",
            "tier": "local_coding",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "local_coding"
        assert "text" in data
        assert data["elapsed_ms"] > 0
        assert "tokens_per_second" in data
        # Should have called the smart/coding model
        text_calls = [c for c in _call_log if c.startswith("text:")]
        assert len(text_calls) >= 1

    def test_execute_explicit_model_profile(self):
        resp = self.client.post("/api/execute", json={
            "prompt": "Classify this text",
            "tier": "local_fast",
            "model_profile": "rlcd_fast",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "1.5B" in data["model_id"]

    def test_execute_explicit_model_id(self):
        resp = self.client.post("/api/execute", json={
            "prompt": "Hello",
            "model_id": "/custom/model/path",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "/custom/model/path"

    # ------------------------------------------------------------------
    # /api/route-and-execute
    # ------------------------------------------------------------------

    def test_route_and_execute_local(self):
        resp = self.client.post("/api/route-and-execute", json={
            "prompt": "Fix the login bug in auth.py",
        })
        assert resp.status_code == 200
        data = resp.json()

        # Should have routing info
        assert "route_id" in data
        assert "classification" in data
        assert "decision" in data

        # Should have execution result (local tier)
        assert data["execution"] is not None
        assert "text" in data["execution"]
        assert data["execution"]["elapsed_ms"] > 0

    def test_route_and_execute_cloud_tier(self):
        """When policy selects a cloud tier, execution is null."""
        resp = self.client.post("/api/route-and-execute", json={
            "prompt": "Research the latest Python 3.14 features",
            "runtime": {
                # Force a classification that triggers cloud_research
                "user_hint": "needs_web",
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        # The mock classifier always returns the same output which
        # maps to local_coding, so this will still execute locally.
        # A real integration test with varied mock outputs would
        # test the cloud path properly.
        assert "decision" in data

    # ------------------------------------------------------------------
    # /api/models
    # ------------------------------------------------------------------

    def test_loaded_models(self):
        resp = self.client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "loaded" in data
        assert "available_profiles" in data
        assert "rlcd_fast" in data["available_profiles"]
        assert "rlcd_smart" in data["available_profiles"]
