from router.escalation import ExecutionState, should_escalate
from router.fallback import recommend_subscription_fallback


def test_recommends_next_eligible_model_after_subscription_exhaustion():
    result = recommend_subscription_fallback(
        eligible_models=("claude_sonnet_4_6", "gpt_5_6_sol", "local_coding"),
        exhausted_model="claude_sonnet_4_6",
    )

    assert result["reason"] == "subscription_exhausted"
    assert result["recommended_model"] == "gpt_5_6_sol"
    assert [item["profile_name"] for item in result["alternatives"]] == ["local_coding"]
    assert result["can_override"] is True


def test_does_not_recommend_the_exhausted_model_or_classifiers():
    result = recommend_subscription_fallback(
        eligible_models=("claude_sonnet_4_6", "rlcd_smart", "local_coding"),
        exhausted_model="claude_sonnet_4_6",
    )

    assert result["recommended_model"] == "local_coding"
    assert all(item["profile_name"] != "rlcd_smart" for item in result["alternatives"])


def test_subscription_exhaustion_is_an_escalation_signal():
    should_move, reason = should_escalate(
        ExecutionState(),
        {"status": "failed", "failure_reason": "subscription_exhausted"},
    )

    assert should_move is True
    assert reason == "subscription_exhausted"


def test_returns_no_recommendation_when_no_models_remain():
    result = recommend_subscription_fallback(
        eligible_models=("claude_sonnet_4_6",),
        exhausted_model="claude_sonnet_4_6",
    )

    assert result["recommended_model"] is None
    assert result["alternatives"] == []
    assert result["can_override"] is True
