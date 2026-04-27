from __future__ import annotations
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file("requirements.txt", remote_path="/app/requirements.txt", copy=True)
    .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml", copy=True)
    .add_local_dir("src", remote_path="/app/src", copy=True)
    .run_commands(
        "pip install --upgrade pip -q",
        "pip install -r /app/requirements.txt -q",
        "pip install -e /app -q",
    )
)

app = modal.App("career-compass-content-engine", image=image)

_secrets = [modal.Secret.from_name("careeer_compass_langchain_content_engine")]

@app.function(
    schedule=modal.Cron("0 12 * * 1-5"),
    secrets=_secrets,
    timeout=900,
)
def run_weekday_workflow():
    from career_compass_content_engine.run_new_blog_workflow import main
    main()

@app.local_entrypoint()
def local_test():
    run_weekday_workflow.remote()
