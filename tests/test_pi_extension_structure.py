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
        'pi.on("tool_call"',
        'pi.on("tool_result"',
        "event.isError",
        "switchAutomaticLocalFailureToCloud",
        "requiresTools",
        "toolCalls",
        "deliverAs: \"followUp\"",

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


def test_extension_preserves_override_until_user_clears_it():
    # An explicit model choice remains active across turns. It is only
    # cleared by the /model-override clear/reset/auto command.
    assert "modelOverride = profileName;" in SOURCE
    assert 'args === "clear"' in SOURCE
    assert 'args === "reset"' in SOURCE
    assert 'args === "auto"' in SOURCE
    assert "Persistent user override" in SOURCE
    # Automatic turn completion must not consume the persistent override.
    assert "modelOverride = null;" in SOURCE  # clear command only


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


def test_extension_resolves_router_dir_through_symlink():
    # The extension is installed as a symlink in ~/.pi/agent/extensions/.
    # Pi reports import.meta.url as the symlink path, so ROUTER_DIR must be
    # derived from the real path or it resolves to ~/.pi/agent and the
    # .venv/bin/python spawn fails with ENOENT.
    assert "realpathSync" in SOURCE
    assert "realpathSync(fileURLToPath(import.meta.url))" in SOURCE
