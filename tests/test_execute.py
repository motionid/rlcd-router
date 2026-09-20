"""
Tests for the /api/execute endpoint and tier-to-model mapping.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        assert next_tier == "local_reasoning"

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


