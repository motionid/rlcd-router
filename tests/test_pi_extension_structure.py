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
