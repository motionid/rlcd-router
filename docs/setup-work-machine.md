# Agent instructions: set up RLCD router on another machine

Use this prompt from Pi (or another coding agent) on the work machine after this repository is available from GitHub. The goal is to install and verify the router using that machine's existing Pi setup—not to change router behavior or copy this machine's private configuration.

## Prompt to give the setup agent

```text
Set up the RLCD turn-aware router from the GitHub repository:
  git@github.com:motionid/rlcd-router.git

Goal: get the router working with this machine's existing Pi installation. Treat this as a machine-specific setup and portability task, not a feature rewrite.

Safety requirements:
- Do not copy .env files, auth files, model stores, API keys, tokens, or other credentials from another machine.
- Never print, log, or include secret values in your report.
- Do not overwrite or remove existing Pi configuration or extensions. Back up any file before editing it, and make the smallest possible change.
- Do not enable paid/metered OpenRouter profiles or make requests that could incur charges. Leave those profiles disabled.
- Keep Ollaya optional and disabled unless I explicitly ask to enable it.
- Do not install a persistent background service, alter login/startup behavior, or change global provider/auth configuration without asking first.
- Do not push commits, change shared repository defaults, or delete user data.
- Ask before any action that could affect other projects, expose a service beyond localhost, or spend money.

Procedure:
1. Inspect the OS/architecture, Python and Node versions, Pi version, current Pi extension layout, and available provider/model registry. Check whether required providers are authenticated without revealing credentials. Report the environment before changing it.
2. Fetch the repository's latest main branch and inspect its README and requirements. Do not assume this machine has the same models or providers as the development machine.
3. Set up an isolated Python virtual environment. Install the documented requirements, first checking whether the MLX dependencies are supported on this OS/architecture. If installation fails because MLX is platform-specific, stop and report the blocker; do not silently remove dependencies or alter requirements. Ask before creating a separate production-only dependency set.
4. Run the Python suite with the repository virtualenv (`.venv/bin/pytest -q`) and the Pi extension suite (`npm run test:extension`). Report failures accurately; do not hide or suppress new warnings to make the test run look clean.
5. Compare every enabled executor profile's exact provider/model pair with this machine's Pi model registry. Identify unavailable pairs and propose local enable/disable or environment-variable overrides. Do not silently edit committed defaults. Ensure that duplicate profiles for the same provider/model do not appear as duplicate automatic candidates.
6. Configure the router to bind only to 127.0.0.1. Use a local, untracked .env file only if needed; create it from documented variables, not from another machine's file. Keep secrets out of it unless genuinely required, and set file permissions appropriately.
7. Start the router in the foreground for initial verification. Check GET /health and GET /api/router/status. Confirm the Ollaya state is disabled unless I separately approved enabling it.
8. Install/link the Pi extension according to the README's multi-file installation instructions. Preserve unrelated files in ~/.pi/agent/extensions. Back up any existing conflicting router link/config before changing it.
9. Reload Pi. Test /router-auto and /router-status, submit one harmless test prompt, and verify the active model is among the router's returned candidates and resolves in Pi. Do not run tasks that could invoke a metered model.
10. Leave the router running only if I ask. Otherwise stop only the process started during this setup. Do not kill a pre-existing router process.
11. Finish with a concise report: repository commit, machine/versions, changes made and backups, exact start/stop and Pi reload commands, test results, provider/model compatibility, and anything blocked or requiring my decision. Do not claim setup is complete until the health/API and Pi activation checks have succeeded.
```

## Notes for the setup agent

- Router defaults are based on the development machine's Pi registry. Exact provider/model identifiers must be checked locally; the router and Pi extension communicate using those identifiers.
- Ollaya is not required. When explicitly enabled, it refines only ambiguous turns and falls back to deterministic classification on failure.
- OpenRouter profiles are intentionally disabled by default. A setup must not activate them without explicit authorization because they can incur per-token charges.
- The router's production routing path is described as not requiring MLX, but `requirements.txt` currently includes MLX packages for legacy/evaluation endpoints. Confirm platform compatibility before attempting non-macOS setup.
