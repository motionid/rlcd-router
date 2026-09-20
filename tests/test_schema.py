"""
Tests for the router schema definition and validation.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.router_schema import (
    ROUTER_SCHEMA,
    CRITICAL_FIELDS,
    ROUTER_FIELD_ORDER,
    validate_router_output,
    build_router_preset,
)


class TestRouterSchema:
    def test_has_expected_fields(self):
        expected = {
            "task_type", "complexity", "coding", "reasoning_depth",
            "fresh_information", "repo_access", "tool_use", "privacy",
            "latency", "failure_cost", "output_contract",
        }
        assert set(ROUTER_SCHEMA.keys()) == expected

    def test_all_fields_are_enum(self):
        for name, spec in ROUTER_SCHEMA.items():
            assert spec["type"] == "enum", f"{name} should be enum"

    def test_all_have_choices(self):
        for name, spec in ROUTER_SCHEMA.items():
            assert "choices" in spec, f"{name} missing choices"
            assert len(spec["choices"]) > 1, f"{name} has only {len(spec['choices'])} choice(s)"

    def test_all_have_descriptions(self):
        for name, spec in ROUTER_SCHEMA.items():
            assert spec.get("description"), f"{name} missing description"

    def test_no_choice_exceeds_255(self):
        for name, spec in ROUTER_SCHEMA.items():
            assert len(spec["choices"]) <= 255, f"{name} exceeds 255 choices"

    def test_field_order_matches_schema(self):
        assert ROUTER_FIELD_ORDER == list(ROUTER_SCHEMA.keys())


class TestCriticalFields:
    def test_critical_fields_exist_in_schema(self):
        for f in CRITICAL_FIELDS:
            assert f in ROUTER_SCHEMA, f"{f} not in schema"

    def test_critical_fields_are_subset(self):
        assert CRITICAL_FIELDS.issubset(set(ROUTER_SCHEMA.keys()))


class TestValidateRouterOutput:
    def test_valid_output(self):
        raw = {
            "task_type": "code_edit",
            "complexity": "low",
            "coding": "dominant",
            "reasoning_depth": "direct",
            "fresh_information": "none",
            "repo_access": "modify",
            "tool_use": "shell",
            "privacy": "ordinary",
            "latency": "normal",
            "failure_cost": "low",
            "output_contract": "exact_patch",
        }
        validated, errors = validate_router_output(raw)
        assert len(errors) == 0
        assert validated == raw

    def test_missing_field(self):
        raw = {"task_type": "code_edit"}
        validated, errors = validate_router_output(raw)
        assert len(errors) == 10  # 10 missing fields
        assert "task_type" in validated

    def test_invalid_choice(self):
        raw = {
            "task_type": "INVALID_VALUE",
            "complexity": "low",
            "coding": "dominant",
            "reasoning_depth": "direct",
            "fresh_information": "none",
            "repo_access": "modify",
            "tool_use": "shell",
            "privacy": "ordinary",
            "latency": "normal",
            "failure_cost": "low",
            "output_contract": "exact_patch",
        }
        validated, errors = validate_router_output(raw)
        assert len(errors) == 1
        assert "task_type" in errors[0]
        assert "task_type" not in validated


class TestBuildRouterPreset:
    def test_preset_structure(self):
        preset = build_router_preset("Fix the login bug")
        assert preset["id"] == "pi_router"
        assert preset["context"] == "Fix the login bug"
        assert preset["schema"] == ROUTER_SCHEMA
