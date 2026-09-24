"""Structural tests for the Pi extension (pi-extension/rlcd-router.ts).

The extension runs inside Pi's TypeScript runtime, which this Python
test environment cannot execute. These tests pin the source structure
instead: the hooks, APIs, and behaviors the router-bundle integration
requires must be present in the extension source.
"""

from pathlib import Path

SOURCE = Path("pi-extension/rlcd-router.ts").read_text()


def test_extension_enforces_and_reports_routes():
    for marker in (
        'pi.on("before_agent_start"',
        'pi.setModel(',
        'pi.on("after_provider_response"',
        'pi.on("agent_before_settle"',
        'POST',
        "/api/route/outcome",
        'failure_reason === "subscription_exhausted"',
        "priorFailures",
        "Fail open",
    ):
        assert marker in SOURCE


def test_extension_skips_non_executors_and_resolves_local_basename():
    assert 'profile.role !== "executor"' in SOURCE
    assert 'provider === "omlx"' in SOURCE
    assert "parts[parts.length - 1]" in SOURCE


def test_extension_consumes_server_billing_order():
    # Task 2 review finding: the override catalog UI must consume the
    # server-reported billing_order instead of hardcoding billing groups
    # (the stale `subscription_unlimited` group no longer exists).
    assert "billing_order" in SOURCE
    assert "subscription_unlimited" not in SOURCE


def test_extension_preserves_one_shot_override_until_activation():
    # Final review finding: a one-shot modelOverride must survive a failed
    # /api/route fetch, unavailable router config, or model activation
    # failure. The shared override may only be cleared after a model was
    # actually activated for the turn, not eagerly when building the
    # request body.
    assert "const overrideForThisTurn = modelOverride;" in SOURCE
    # Consumption is gated on the turn-local capture after activation.
    assert "if (overrideForThisTurn) {" in SOURCE
    # The eager, unconditional consumption pattern is gone.
    assert "modelOverride = null; // one-shot, consumed here" not in SOURCE


def test_extension_clears_route_id_when_no_model_activated():
    # Final review finding: lastRouteId must be cleared on every failure
    # path after the route decision arrives, so agent_before_settle never
    # reports an outcome for a route that never activated a model.
    failure_markers = (
        "Router config unavailable",
        "No available model for tier",
        "Switch failed",
    )
    for marker in failure_markers:
        idx = SOURCE.index(marker)
        window = SOURCE[max(0, idx - 400) : idx]
        assert "lastRouteId = null" in window, (
            f"failure branch '{marker}' must clear lastRouteId before returning"
        )
    # The unreachable-router catch path must also drop a pending route id.
    catch_idx = SOURCE.index("Router unreachable")
    assert "lastRouteId = null" in SOURCE[max(0, catch_idx - 400) : catch_idx]
