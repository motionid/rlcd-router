"""Structural tests for the turn-aware Pi extension (pi-extension/turn-router.ts).

The extension runs inside Pi's TypeScript runtime, which this Python test
environment cannot execute.  These tests pin the source structure instead:
the documented lifecycle hooks, state/model APIs, and server endpoints the
turn-aware router requires must be present in the extension source.

The source is read lazily inside a fixture so a missing extension file
fails the tests instead of breaking pytest collection (the stale
collection failure of the previous rlcd-router.ts test module).
"""

from pathlib import Path

import pytest

SOURCE_PATH = Path("pi-extension") / "turn-router.ts"
DELETED_EXTENSION = "rlcd-router.ts"


@pytest.fixture(scope="module")
def source():
    """The extension source, read lazily so absence fails tests, not collection."""
    if not SOURCE_PATH.exists():
        pytest.fail(f"extension source is missing: {SOURCE_PATH}")
    return SOURCE_PATH.read_text()


def test_extension_uses_documented_lifecycle_hooks(source):
    for marker in (
        'pi.on("session_start"',
        'pi.on("before_agent_start"',
        'pi.on("after_provider_response"',
        'pi.on("agent_before_settle"',
        'pi.on("model_select"',
        'pi.on("session_shutdown"',
    ):
        assert marker in source


def test_extension_uses_state_and_model_apis(source):
    # Persistence goes through appendEntry; branch-sensitive state is
    # rebuilt from ctx.sessionManager.getBranch(); candidates are
    # resolved through the PI model registry.
    for marker in (
        "pi.appendEntry(",
        "pi.setModel(",
        "ctx.sessionManager.getBranch",
        "ctx.modelRegistry",
    ):
        assert marker in source


def test_extension_registers_manual_controls(source):
    for marker in (
        'pi.registerCommand("router-auto"',
        'pi.registerCommand("router-status"',
    ):
        assert marker in source


def test_extension_consumes_server_route_and_status_apis(source):
    for marker in (
        "/api/route",
        "/api/route/outcome",
        "/api/router/status",
        "/health",
    ):
        assert marker in source


def test_extension_never_references_deleted_extension_or_rlcd_tools(source):
    assert DELETED_EXTENSION not in source
    # The turn router must not call the RLCD classification/execution
    # tools: routing decisions come from the /api/route contract only.
    for forbidden in ("/api/run-rlcd", "/api/route-and-execute", "/api/execute"):
        assert forbidden not in source


def test_automatic_setModel_is_guarded(source):
    # An automatic setModel (candidate activation) must not create a
    # user override: model_select only records an override when the
    # internal guard token is not held, and the token wraps pi.setModel.
    assert "automaticSwitchToken" in source
    guard_assignment = source.index("automaticSwitchToken = true")
    set_model_call = source.index("pi.setModel(")
    assert guard_assignment < set_model_call
    model_select = source.index('pi.on("model_select"')
    assert source.index("automaticSwitchToken", model_select) > model_select


def test_manual_selection_persists_until_router_auto(source):
    # A user model selection is a persistent override; only /router-auto
    # clears it, and before_agent_start must respect it.
    assert "clearOverride" in source
    assert "state.overrideModel" in source


def test_ui_is_guarded_for_non_ui_modes(source):
    # Normal route UI is one compact line, guarded for non-UI modes.
    assert 'ctx.mode === "tui"' in source
    assert "ctx.hasUI" in source
    assert "ROUTE " in source


def test_extension_fails_open_and_never_blocks_a_turn(source):
    # Router outage / outcome reporting failures retain the active model
    # and never block the turn.
    assert "outageDecision" in source
    catch_count = source.count("catch")
    assert catch_count >= 3


def test_server_autostart_never_runs_in_factory(source):
    # Long-lived resources (optional router auto-start) belong to
    # session_start; the factory registers handlers only.
    factory = source.index("export default")
    first_handler = source.index("pi.on(")
    factory_body = source[factory:first_handler]
    assert "spawn(" not in factory_body
    assert "exec(" not in factory_body
    # The only process spawn must live after the session_start registration.
    assert source.index("spawn(") > source.index('pi.on("session_start"')


def test_continuation_is_bounded(source):
    # agent_before_settle continues only when a next candidate was
    # activated and the per-route attempt limit permits it.
    assert "maxAttempts" in source
    assert "continue: true" in source
