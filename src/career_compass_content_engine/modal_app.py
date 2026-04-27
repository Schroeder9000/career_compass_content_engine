from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Container image — bakes in all deps and the package source at deploy time.
# Run `modal deploy src/career_compass_content_engine/modal_app.py` from the
# workspace root to rebuild and redeploy.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .copy_local_file("requirements.txt", "/app/requirements.txt")
    .copy_local_file("pyproject.toml", "/app/pyproject.toml")
    .copy_local_dir("src", "/app/src")
    .run_commands(
        "pip install --upgrade pip -q",
        "pip install -r /app/requirements.txt -q",
        "pip install -e /app -q",
    )
)

app = modal.App("career-compass-content-engine", image=image)

# ---------------------------------------------------------------------------
# Secrets — loads all env vars from your Modal secret into the container.
# Update the name below if your secret has a different name.
# ---------------------------------------------------------------------------
_secrets = [modal.Secret.from_name("careeer_compass_langchain_content_engine")]


@app.function(
    schedule=modal.Cron("0 12 * * 1-5"),  # 7 AM ET (UTC-5 / EDT) Mon–Fri
    secrets=_secrets,
    timeout=900,  # 15 min — covers image gen + full LLM pipeline
)
def run_weekday_workflow():  # pragma: no cover - executed by Modal
    from career_compass_content_engine.run_new_blog_workflow import main
    main()


@app.local_entrypoint()
def local_test():
    """Trigger one remote run for smoke-testing.

    Usage:
        modal run src/career_compass_content_engine/modal_app.py
    """
    run_weekday_workflow.remote()  # pragma: no cover
