"""
Integration test: exercise /api/route, /api/execute, and
/api/route-and-execute without loading MLX models.

Uses the TestClient from FastAPI with a mock RLCD engine.
"""

from __future__ import annotations

import sys
import os
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from router.availability import FailureCategory
from router.config import MODEL_PROFILES, AccessClass, Capability


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
                "claude_sonnet_5",
                "gpt_5_6_sol",
                "local_coding",
            ]
        )
        app_module._telemetry_log.record(event)

        resp = self.client.post("/api/route/outcome", json={
            "event_id": event.event_id,
            "status": "failed",
            "failure_reason": "subscription_exhausted",
            "actual_model": "claude_sonnet_5",
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

    def test_route_config_exposes_catalog_invariants(self):
        import router.config as config
        assert config.MODEL_PROFILES["local_coding"].role == "executor"
        assert config.MODEL_PROFILES["rlcd_smart"].role == "classifier"
        assert config.MODEL_PROFILES["local_coding"].billing == "local"

    def test_route_config_serializes_capability_and_access_class(self):
        resp = self.client.get("/api/route/config")
        assert resp.status_code == 200
        models = resp.json()["models"]

        local_coding = models["local_coding"]
        assert local_coding["capability"] == "strong"
        assert local_coding["access_class"] == "local"
        assert local_coding["billing"] == "local"  # legacy compat retained

        opus = models["claude_opus_5"]
        assert opus["capability"] == "reasoning"
        assert opus["access_class"] == "subscription"
        assert opus["billing"] == "subscription_quota"  # legacy compat retained

        for name, model in models.items():
            assert model["capability"] in {
                "basic", "normal", "strong", "reasoning",
            }, name
            assert model["access_class"] in {
                "local", "subscription", "metered-cheap", "metered-expensive",
            }, name

    def test_route_decision_exposes_capability_and_candidate_fields(self):
        resp = self.client.post("/api/route", json={
            "prompt": "Fix the login bug in auth.py",
        })
        assert resp.status_code == 200
        decision = resp.json()["decision"]

        assert decision["capability"] in {
            "basic", "normal", "strong", "reasoning",
        }
        # Task 5's policy now populates candidates on the production path.
        assert isinstance(decision["candidates"], list)
        assert decision["candidates"]
        assert isinstance(decision["excluded"], list)
        for entry in decision["candidates"]:
            assert entry["capability"] in {
                "basic", "normal", "strong", "reasoning",
            }
            assert entry["access_class"] in {
                "local", "subscription", "metered-cheap", "metered-expensive",
            }
        for entry in decision["excluded"]:
            assert isinstance(entry["reason"], str)
        # Compatibility fields retained until migration completes.
        assert decision["eligible_models"] == [
            c["profile_id"] for c in decision["candidates"]
        ]

    def test_override_catalog_uses_supported_billing_order(self):
        response = self.client.get("/api/models/overrides")
        assert response.status_code == 200
        body = response.json()
        profile_names = {
            model["profile_name"]
            for models in body["models"].values()
            for model in models
        }
        assert "rlcd_fast" not in profile_names
        assert "rlcd_smart" not in profile_names
        assert body["billing_order"] == ["local", "subscription_quota", "paid"]

    def test_route_basic(self):
        resp = self.client.post("/api/route", json={
            "prompt": "Fix the login bug in auth.py",
        })
        assert resp.status_code == 200
        data = resp.json()

        assert "route_id" in data
        assert "classification" in data
        assert "decision" in data

        # Production path: deterministic classification, never the RLCD
        # engine (which stays patched in this class to prove it is
        # never invoked).
        cls = data["classification"]
        assert cls["classifier"] == "deterministic"
        assert cls["capability"] == "normal"
        assert not [c for c in _call_log if c.startswith("rlcd:")]

        decision = data["decision"]
        assert decision["capability"] == "normal"
        assert decision["candidates"]
        assert decision["tier"] == "cloud_standard"

    def test_route_with_runtime_prior_failures(self):
        resp = self.client.post("/api/route", json={
            "prompt": "Fix the login bug",
            "runtime": {"prior_failures": 1},
        })
        assert resp.status_code == 200
        data = resp.json()
        # Prior failures on basic coding → cloud_standard
        assert data["decision"]["tier"] == "cloud_standard"

    def test_route_bypass(self):
        """options.bypass skips classification: conservative, subscription-safe."""
        resp = self.client.post("/api/route", json={
            "prompt": "Hello world",
            "options": {"bypass": True},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"]["classifier"] == "bypass"
        assert data["classification"]["ambiguous"] is True
        # No models, no network, no RLCD: conservative subscription-first
        # route with paid access excluded by uncertainty.
        candidates = data["decision"]["candidates"]
        assert candidates
        assert candidates[0]["access_class"] == "subscription"
        assert all(
            c["access_class"] in ("subscription", "local")
            for c in candidates
        )
        assert _call_log == []

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
        """Production routing never consults the RLCD cascade."""
        resp = self.client.post("/api/route", json={
            "prompt": "Fix the login bug in auth.py",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"]["classifier"] == "deterministic"
        # The engine is patched in this class; nothing may call it.
        assert _call_log == []

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


# ===================================================================
# Production router API — deterministic route, outcome, and status
#
# These tests deliberately do NOT patch or import the RLCD engine:
# the production path must work without MLX and without network.
# ===================================================================

class TestProductionRouterAPI:
    def setup_method(self):
        import importlib
        import server.app as app_module
        self.app_module = importlib.reload(app_module)
        self.client = TestClient(self.app_module.app)

    def _route(self, prompt, **extra):
        payload = {"prompt": prompt}
        payload.update(extra)
        return self.client.post("/api/route", json=payload)

    # ------------------------------------------------------------------
    # Production /api/route
    # ------------------------------------------------------------------

    def test_route_obvious_request_does_not_load_rlcd(self):
        """Subprocess proof: an obvious route never imports core.engine."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys, os\n"
            "sys.path.insert(0, os.getcwd())\n"
            "from fastapi.testclient import TestClient\n"
            "import server.app\n"
            "client = TestClient(server.app.app)\n"
            "resp = client.post('/api/route', json={\n"
            "    'prompt': 'rename the variable foo to bar'})\n"
            "assert resp.status_code == 200, resp.text\n"
            "body = resp.json()\n"
            "assert body['classification']['classifier'] == 'deterministic'\n"
            "for module in ('core.engine', 'core.engine_mlx', 'mlx'):\n"
            "    assert module not in sys.modules, module\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_production_route_succeeds_when_mlx_imports_raise(self):
        """Task 10: with every MLX import/call site patched to raise,
        /api/route still succeeds and never builds the RLCD classifier."""
        app = self.app_module
        blocked = ("core", "core.engine", "core.engine_mlx", "core.schema",
                   "mlx", "mlx_lm")
        saved_modules = {name: sys.modules.get(name, None) for name in blocked}

        class _ImportBlocker:
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split(".")[0]
                if root in ("core", "mlx", "mlx_lm"):
                    raise AssertionError(
                        f"production routing must not import {fullname}"
                    )
                return None

        blocker = _ImportBlocker()
        sys.meta_path.insert(0, blocker)
        # Poison any already-imported backend modules too.
        for name in blocked:
            sys.modules[name] = None

        def _fail():
            raise AssertionError(
                "production routing must not build the RLCD classifier"
            )

        saved_builder = app._get_router_classifier
        app._get_router_classifier = _fail
        try:
            resp = self._route("Fix the login bug in auth.py")
        finally:
            sys.meta_path.remove(blocker)
            app._get_router_classifier = saved_builder
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        assert resp.status_code == 200
        body = resp.json()
        assert body["classification"]["classifier"] == "deterministic"
        assert body["decision"]["candidates"]

    def test_route_ambiguous_request_works_with_ollaya_disabled(self):
        resp = self._route("hi")
        assert resp.status_code == 200
        body = resp.json()
        cls = body["classification"]
        assert cls["classifier"] == "deterministic"
        assert cls["ambiguous"] is True
        assert cls["capability"] == "normal"
        candidates = body["decision"]["candidates"]
        assert candidates
        # Ambiguity never selects paid access; subscription first.
        assert all(
            c["access_class"] in ("subscription", "local")
            for c in candidates
        )
        assert candidates[0]["access_class"] == "subscription"
        assert body["decision"]["tier"] == "cloud_standard"

    def test_route_returns_ordered_candidates_and_exclusions(self):
        resp = self._route(
            "Debug the failing test suite across the repository",
            requirements={"supports_tools": True},
        )
        assert resp.status_code == 200
        assert resp.json()["classification"]["classifier"] == "deterministic"
        decision = resp.json()["decision"]
        candidates = decision["candidates"]
        assert candidates

        # Ordering: access-class rank, then priority, then name.
        eligible = [
            (name, profile)
            for name, profile in MODEL_PROFILES.items()
            if profile.role == "executor" and profile.enabled
            and profile.capability >= Capability.STRONG
            and profile.supports_tools
            and profile.access_class is AccessClass.SUBSCRIPTION
        ]
        expected = [
            name for name, _ in
            sorted(eligible, key=lambda item: (item[1].priority, item[0]))
        ]
        assert [c["profile_id"] for c in candidates] == expected

        excluded = {e["profile_id"]: e["reason"] for e in decision["excluded"]}
        assert excluded["rlcd_smart"] == "classifier_profile"
        assert excluded["rlcd_fast"] == "classifier_profile"
        assert excluded["local_coding"] == "access_class_excluded"
        assert excluded["gpt_6_luna"] == "insufficient_capability"

    def test_route_local_only_requirement_returns_local_candidates_only(self):
        resp = self._route(
            "Summarize this output for me",
            requirements={"local_only": True},
        )
        assert resp.status_code == 200
        decision = resp.json()["decision"]
        candidates = decision["candidates"]
        assert candidates
        assert all(c["access_class"] == "local" for c in candidates)
        assert candidates[0]["profile_id"] == "local_coding"

    def test_uncertain_route_with_no_eligible_candidates_returns_empty(self):
        store = self.app_module._get_availability_store()
        for name, profile in MODEL_PROFILES.items():
            if profile.access_class is AccessClass.LOCAL:
                store.record_failure(name, FailureCategory.CAPACITY)
        resp = self._route("hi", requirements={"local_only": True})
        assert resp.status_code == 200
        decision = resp.json()["decision"]
        assert decision["candidates"] == []
        assert decision["eligible_models"] == []
        assert "no_eligible_candidates" in decision["reasons"]

    def test_router_exception_returns_safe_fallback_decision(self):
        app = self.app_module

        def boom(prompt, context, ollaya=None):
            raise RuntimeError("classifier exploded")

        original = getattr(app, "classify_task", None)
        app.classify_task = boom
        try:
            resp = self._route("Fix the login bug")
        finally:
            if original is not None:
                app.classify_task = original

        assert resp.status_code == 200
        body = resp.json()
        decision = body["decision"]
        assert decision["classifier_used"] == "fallback"
        assert "internal_error" in decision["reasons"]
        candidates = decision["candidates"]
        assert candidates
        # A real, configured, eligible subscription profile — never paid,
        # never fabricated.
        assert all(
            c["access_class"] == "subscription" for c in candidates
        )
        assert body["classification"]["classifier"] == "fallback"

    def test_no_network_classifier_is_required(self):
        import urllib.request

        def no_network(*args, **kwargs):
            raise AssertionError("production routing must not touch the network")

        original = urllib.request.urlopen
        urllib.request.urlopen = no_network
        try:
            resp = self._route("hi")
        finally:
            urllib.request.urlopen = original
        assert resp.status_code == 200
        assert resp.json()["classification"]["classifier"] == "deterministic"

    # ------------------------------------------------------------------
    # Explicit overrides
    # ------------------------------------------------------------------

    def test_explicit_override_returns_only_requested_available_profile(self):
        resp = self._route(
            "Review this migration plan",
            model_override="gpt_5_6_terra",
        )
        assert resp.status_code == 200
        body = resp.json()
        candidates = body["decision"]["candidates"]
        assert len(candidates) == 1
        assert candidates[0]["profile_id"] == "gpt_5_6_terra"
        assert candidates[0]["access_class"] == "subscription"
        assert body["decision"]["eligible_models"] == ["gpt_5_6_terra"]
        assert body["decision"]["classifier_used"] == "user_model_override"
        assert body["override"] == {
            "requested": "gpt_5_6_terra",
            "available": True,
        }

    def test_unavailable_override_falls_back_without_clearing_preference(self):
        store = self.app_module._get_availability_store()
        store.record_failure("gpt_5_6_terra", FailureCategory.QUOTA)

        resp = self._route(
            "Review this migration plan",
            model_override="gpt_5_6_terra",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["override"] == {
            "requested": "gpt_5_6_terra",
            "available": False,
        }
        ids = [c["profile_id"] for c in body["decision"]["candidates"]]
        assert ids
        assert "gpt_5_6_terra" not in ids
        assert "override_unavailable:gpt_5_6_terra" in body["decision"]["reasons"]
        # No server-side preference clear: the profile stays in cooldown
        # untouched, so a future turn can retry the override.
        assert store.is_available("gpt_5_6_terra") is False
        assert store.snapshot()["gpt_5_6_terra"].reason is FailureCategory.QUOTA

    # ------------------------------------------------------------------
    # /api/route/outcome
    # ------------------------------------------------------------------

    def _first_route_candidate(self, prompt="Fix the login bug in auth.py"):
        resp = self._route(prompt)
        assert resp.status_code == 200
        body = resp.json()
        candidates = body["decision"]["candidates"]
        assert candidates
        return body["route_id"], candidates[0]["profile_id"]

    def test_outcome_rate_limit_updates_availability_not_task_capability(self):
        route_id, profile_id = self._first_route_candidate()

        resp = self.client.post("/api/route/outcome", json={
            "event_id": route_id,
            "status": "failed",
            "failure_category": "rate_limit",
            "candidate_profile": profile_id,
            "retry_after": 30.0,
        })
        assert resp.status_code == 200
        availability = resp.json()["availability"]
        assert availability["status"] == "cooldown"
        assert availability["reason"] == "rate_limit"
        assert availability["retry_after"] == 30.0

        # Provider failure cools the profile down; it must not escalate
        # the task capability.
        resp2 = self._route("Fix the login bug in auth.py")
        body2 = resp2.json()
        assert body2["decision"]["capability"] == "normal"
        ids = [c["profile_id"] for c in body2["decision"]["candidates"]]
        assert profile_id not in ids
        excluded = {
            e["profile_id"]: e["reason"] for e in body2["decision"]["excluded"]
        }
        assert excluded[profile_id] == "unavailable_cooldown"

    def test_outcome_task_failure_updates_task_failure_not_availability(self):
        route_id, profile_id = self._first_route_candidate()

        resp = self.client.post("/api/route/outcome", json={
            "event_id": route_id,
            "status": "failed",
            "failure_category": "task_failure",
            "candidate_profile": profile_id,
        })
        assert resp.status_code == 200
        out = resp.json()
        availability = out["availability"]
        # The provider was available; the work failed. No cooldown.
        assert availability["status"] == "available"
        assert availability["cooldown_until"] is None
        assert availability["failure_count"] == 1
        # Telemetry records the failed attempt on the route's single
        # record.  Task 8: a failed non-final candidate keeps the record
        # pending (it settles on success or candidate exhaustion).
        ev = self.app_module._telemetry_log.get_event(route_id)
        assert ev is not None
        assert ev.attempts and ev.attempts[-1].status == "failed"
        assert ev.attempts[-1].failure_category == "task_failure"
        # The profile stays eligible on the next turn.
        resp2 = self._route("Fix the login bug in auth.py")
        ids = [
            c["profile_id"]
            for c in resp2.json()["decision"]["candidates"]
        ]
        assert profile_id in ids

    def test_outcome_success_clears_cooldown(self):
        route_id, profile_id = self._first_route_candidate()
        store = self.app_module._get_availability_store()
        store.record_failure(profile_id, FailureCategory.RATE_LIMIT)
        assert store.is_available(profile_id) is False

        resp = self.client.post("/api/route/outcome", json={
            "event_id": route_id,
            "status": "success",
            "candidate_profile": profile_id,
        })
        assert resp.status_code == 200
        assert resp.json()["availability"]["status"] == "available"
        assert store.is_available(profile_id) is True

    def test_outcome_invalid_failure_category_is_machine_readable_error(self):
        route_id, profile_id = self._first_route_candidate()
        resp = self.client.post("/api/route/outcome", json={
            "event_id": route_id,
            "status": "failed",
            "failure_category": "not_a_category",
            "candidate_profile": profile_id,
        })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error"] == "invalid_failure_category"
        assert detail["value"] == "not_a_category"
        assert "rate_limit" in detail["valid_categories"]
        # Nothing was recorded: the store stays untouched.
        store = self.app_module._get_availability_store()
        assert store.is_available(profile_id) is True

    # ------------------------------------------------------------------
    # /api/router/status
    # ------------------------------------------------------------------

    def test_status_reports_classifier_and_cooldowns(self):
        resp = self.client.get("/api/router/status")
        assert resp.status_code == 200
        body = resp.json()

        assert body["classifier"]["deterministic"] is True
        ollaya = body["classifier"]["ollaya"]
        assert ollaya["enabled"] is False
        assert ollaya["health"] == "disabled"

        assert "local_coding" in body["models"]
        local_coding = body["models"]["local_coding"]
        assert local_coding["capability"] == "strong"
        assert local_coding["access_class"] == "local"
        assert local_coding["availability"]["status"] == "available"

        # No route yet: last_route is null.
        assert body["last_route"] is None

        # A cooldown shows up in the status snapshot.
        store = self.app_module._get_availability_store()
        store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)
        resp2 = self.client.get("/api/router/status")
        avail = resp2.json()["models"]["gpt_6_luna"]["availability"]
        assert avail["status"] == "cooldown"
        assert avail["reason"] == "rate_limit"
        assert avail["cooldown_until"] is not None

        # The last route is reported compactly.
        self._route("Fix the login bug in auth.py")
        resp3 = self.client.get("/api/router/status")
        last = resp3.json()["last_route"]
        assert last["classifier"] == "deterministic"
        assert last["capability"] == "normal"
        assert isinstance(last["candidates"], list)
        assert last["candidates"]

    def test_status_availability_entries_are_plain_copies(self):
        app = self.app_module
        store = app._get_availability_store()
        store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)
        snapshot = app._availability_snapshot()
        entry = snapshot["gpt_6_luna"]
        assert type(entry) is dict
        assert entry == store.snapshot()["gpt_6_luna"].to_dict()
        # Mutating the serialized copy must never touch live state.
        entry["status"] = "available"
        assert store.snapshot()["gpt_6_luna"].status == "cooldown"

    # ------------------------------------------------------------------
    # Fix round 1: override compatibility under caller requirements
    # ------------------------------------------------------------------

    def test_local_only_cloud_override_returns_no_cloud_candidate(self):
        """Privacy regression: local_only + cloud override never yields cloud."""
        resp = self._route(
            "Summarize this output for me",
            model_override="gpt_5_6_terra",
            requirements={"local_only": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # The override preference is preserved in metadata, not cleared.
        assert body["override"]["requested"] == "gpt_5_6_terra"
        assert body["override"]["available"] is True
        assert body["override"]["compatible"] is False
        assert body["override"]["reason"] == "local_only_blocks_remote"
        assert (
            "override_incompatible:local_only_blocks_remote"
            in body["decision"]["reasons"]
        )
        candidates = body["decision"]["candidates"]
        assert candidates
        assert all(c["access_class"] == "local" for c in candidates)
        assert "gpt_5_6_terra" not in [
            c["profile_id"] for c in candidates
        ]

    def test_override_incompatible_with_tools_reports_stable_reason(self):
        resp = self._route(
            "Fix the login bug in auth.py",
            model_override="local_general",   # supports_tools=False
            requirements={"supports_tools": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["override"] == {
            "requested": "local_general",
            "available": True,
            "compatible": False,
            "reason": "tools_unsupported",
        }
        assert "override_incompatible:tools_unsupported" in body["decision"]["reasons"]
        ids = [c["profile_id"] for c in body["decision"]["candidates"]]
        assert ids
        assert "local_general" not in ids

    def test_compatible_override_metadata_preserved(self):
        resp = self._route(
            "Review this migration plan",
            model_override="gpt_5_6_terra",
            requirements={"supports_tools": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["override"] == {
            "requested": "gpt_5_6_terra",
            "available": True,
        }
        assert len(body["decision"]["candidates"]) == 1
        assert body["decision"]["candidates"][0]["profile_id"] == "gpt_5_6_terra"

    # ------------------------------------------------------------------
    # Fix round 1: outcome handling independent of telemetry
    # ------------------------------------------------------------------

    def test_outcome_updates_availability_when_telemetry_fails(self):
        app = self.app_module

        def boom_start(*args, **kwargs):
            raise RuntimeError("telemetry backend down")

        def boom_outcome(*args, **kwargs):
            raise RuntimeError("telemetry backend down")

        app._telemetry_log.start_event = boom_start
        app._telemetry_log.report_outcome = boom_outcome
        resp = self._route("Fix the login bug in auth.py")
        assert resp.status_code == 200  # telemetry failure never breaks routing
        body = resp.json()
        route_id = body["route_id"]
        profile_id = body["decision"]["candidates"][0]["profile_id"]

        out = self.client.post("/api/route/outcome", json={
            "event_id": route_id,
            "status": "failed",
            "failure_category": "rate_limit",
            "candidate_profile": profile_id,
        })
        assert out.status_code == 200
        assert out.json()["availability"]["status"] == "cooldown"
        assert out.json()["availability"]["reason"] == "rate_limit"

        # The cooldown is visible to the next route even though telemetry
        # never recorded the event.
        resp2 = self._route("Fix the login bug in auth.py")
        ids = [
            c["profile_id"] for c in resp2.json()["decision"]["candidates"]
        ]
        assert profile_id not in ids

    # ------------------------------------------------------------------
    # Task 8: single-record telemetry settlement
    # ------------------------------------------------------------------

    def test_route_outcome_finalizes_single_record_with_attempts(self):
        app = self.app_module
        resp = self._route(
            "Debug the race condition across the repository"
        )
        route_id = resp.json()["route_id"]
        profile_id = resp.json()["decision"]["candidates"][0]["profile_id"]

        # Pending: in memory, not finalized, not in aggregates.
        assert app._telemetry_log.get_event(route_id).status == "pending"
        assert route_id not in [e.event_id for e in app._telemetry_log.events]

        out = self.client.post("/api/route/outcome", json={
            "event_id": route_id,
            "status": "success",
            "candidate_profile": profile_id,
            "latency_ms": 1000.0,
        })
        assert out.status_code == 200

        events = [e for e in app._telemetry_log.events
                  if e.event_id == route_id]
        assert len(events) == 1  # exactly one finalized record
        ev = events[0]
        assert ev.status == "success"
        assert len(ev.attempts) == 1
        assert ev.attempts[0].profile_id == profile_id

        # A repeated outcome report for the same route/attempt is an
        # idempotent no-op: no duplicate record, no double-counted cost.
        out2 = self.client.post("/api/route/outcome", json={
            "event_id": route_id,
            "status": "success",
            "candidate_profile": profile_id,
            "latency_ms": 1000.0,
            "cost": 0.01,
        })
        assert out2.status_code == 200
        events = [e for e in app._telemetry_log.events
                  if e.event_id == route_id]
        assert len(events) == 1
        assert len(events[0].attempts) == 1
        assert events[0].latency_ms == 1000.0

    def test_route_outcome_candidate_exhaustion_finalizes_failed(self):
        app = self.app_module
        # local_only keeps the candidate set small and deterministic.
        resp = self._route(
            "Summarize this output for me",
            requirements={"local_only": True},
        )
        route_id = resp.json()["route_id"]
        candidates = resp.json()["decision"]["candidates"]
        assert candidates
        last = candidates[-1]["profile_id"]

        out = self.client.post("/api/route/outcome", json={
            "event_id": route_id,
            "status": "failed",
            "failure_category": "task_failure",
            "candidate_profile": last,
        })
        assert out.status_code == 200
        ev = app._telemetry_log.get_event(route_id)
        assert ev.status == "failed"  # exhausted: settled as failed
        assert len(app._telemetry_log.events) >= 1

    def test_route_index_is_bounded_with_oldest_eviction(self):
        app = self.app_module
        index = app._route_index
        index.clear()
        for i in range(1030):
            app._register_route_index(f"route-{i}", ["gpt_6_luna"])
        assert app._ROUTE_INDEX_MAX == 1024
        assert len(index) == 1024
        # Deterministic oldest-first eviction: the first six are gone,
        # the newest survive.
        assert "route-0" not in index
        assert "route-5" not in index
        assert "route-6" in index
        assert "route-1029" in index

    # ------------------------------------------------------------------
    # Fix round 1: safe fallback capability validation
    # ------------------------------------------------------------------

    def _route_with_broken_policy(self, prompt, **extra):
        app = self.app_module

        def boom(*args, **kwargs):
            raise RuntimeError("policy exploded")

        original = app.build_route
        app.build_route = boom
        try:
            return self._route(prompt, **extra)
        finally:
            app.build_route = original

    def test_fallback_respects_minimum_task_capability(self):
        resp = self._route_with_broken_policy(
            "Prove the correctness of the invariant with a formal proof "
            "and a derivation of the bound"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["classification"]["capability"] == "reasoning"
        decision = body["decision"]
        assert decision["classifier_used"] == "fallback"
        candidates = decision["candidates"]
        assert candidates
        assert all(
            c["capability"] in ("reasoning",) for c in candidates
        )

    def test_fallback_capability_rejection_is_terminal(self):
        store = self.app_module._get_availability_store()
        for name, profile in MODEL_PROFILES.items():
            if (
                profile.access_class is AccessClass.SUBSCRIPTION
                and profile.capability >= Capability.REASONING
            ):
                store.record_failure(name, FailureCategory.QUOTA)
        resp = self._route_with_broken_policy(
            "Prove the correctness of the invariant with a formal proof"
        )
        assert resp.status_code == 200
        decision = resp.json()["decision"]
        assert decision["candidates"] == []
        assert decision["classifier_used"] == "fallback"
        assert "no_eligible_candidates" in decision["reasons"]

    def test_fallback_rejects_thinking_requirement_without_capable_profiles(self):
        store = self.app_module._get_availability_store()
        for name, profile in MODEL_PROFILES.items():
            if profile.supports_thinking:
                store.record_failure(name, FailureCategory.QUOTA)
        resp = self._route_with_broken_policy(
            "Fix the login bug in auth.py",
            requirements={"supports_thinking": True},
        )
        assert resp.status_code == 200
        decision = resp.json()["decision"]
        assert decision["candidates"] == []
        assert "no_eligible_candidates" in decision["reasons"]

    def test_fallback_rejects_unsatisfiable_context_requirement(self):
        resp = self._route_with_broken_policy(
            "Fix the login bug in auth.py",
            requirements={"minimum_context": 2_000_000},
        )
        assert resp.status_code == 200
        decision = resp.json()["decision"]
        assert decision["candidates"] == []
        assert "no_eligible_candidates" in decision["reasons"]
