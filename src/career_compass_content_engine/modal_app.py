from __future__ import annotations

import modal

from .run_new_blog_workflow import main as run_content_workflow


app = modal.App("career-compass-content-engine")


@app.function(schedule=modal.Period(days=7))
def weekly_workflow():  # pragma: no cover - executed by Modal
    run_content_workflow()
