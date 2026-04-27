from __future__ import annotations
import os
from datetime import datetime, timezone

import modal
import requests

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


def _send_email_alert(subject: str, body: str) -> bool:
    """Send an email alert via MailerSend (or Resend fallback)."""
    to_email = os.getenv("ALERT_TO_EMAIL", "datadave@careeraiapp.com")
    from_email = os.getenv("ALERT_FROM_EMAIL", "alerts@careeraiapp.com")

    # Preferred provider: MailerSend
    api_key = os.getenv("MAILERSEND_API_KEY")
    if api_key:
        payload = {
            "from": {"email": from_email, "name": "Career Compass Alerts"},
            "to": [{"email": to_email}],
            "subject": subject,
            "text": body,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            response = requests.post(
                "https://api.mailersend.com/v1/email",
                headers=headers,
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            print(f"[alerts] MailerSend alert failed: {exc}")

    # Fallback provider: Resend
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("[alerts] MAILERSEND_API_KEY/RESEND_API_KEY not set; skipping email alert")
        return False

    payload = {
        "from": from_email,
        "to": [to_email],
        "subject": subject,
        "text": body,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers=headers,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        print(f"[alerts] email alert failed: {exc}")
        return False


def _send_webhook_alert(subject: str, body: str, ok: bool) -> bool:
    """Optional push-style notification via generic webhook."""
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if not webhook_url:
        return False

    payload = {
        "status": "success" if ok else "failure",
        "subject": subject,
        "message": body,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        response = requests.post(webhook_url, json=payload, timeout=20)
        response.raise_for_status()
        return True
    except Exception as exc:
        print(f"[alerts] webhook alert failed: {exc}")
        return False


def _notify(subject: str, body: str, ok: bool) -> None:
    email_sent = _send_email_alert(subject, body)
    webhook_sent = _send_webhook_alert(subject, body, ok)
    print(f"[alerts] sent: email={email_sent} webhook={webhook_sent}")

@app.function(
    schedule=modal.Cron("0 12 * * 1-5"),
    secrets=_secrets,
    timeout=900,
)
def run_weekday_workflow():
    from career_compass_content_engine.run_new_blog_workflow import main

    started_at = datetime.now(timezone.utc)
    try:
        result = main()
        publish_result = result.publish_result or {}
        metadata = publish_result.get("metadata", {})
        title = metadata.get("title") or metadata.get("name") or "(title unavailable)"
        item_id = (
            publish_result.get("webflow_item_id")
            or publish_result.get("webflow_item", {}).get("id")
            or "(item id unavailable)"
        )
        subject = "Career Compass workflow success"
        body = (
            f"Status: SUCCESS\n"
            f"Started (UTC): {started_at.isoformat()}\n"
            f"Finished (UTC): {datetime.now(timezone.utc).isoformat()}\n"
            f"Title: {title}\n"
            f"Webflow Item ID: {item_id}\n"
            f"Total tokens: {result.total_tokens}\n"
            f"Estimated cost USD: {result.total_cost_usd:.4f}\n"
        )
        _notify(subject, body, ok=True)
    except Exception as exc:
        subject = "Career Compass workflow failure"
        body = (
            f"Status: FAILURE\n"
            f"Started (UTC): {started_at.isoformat()}\n"
            f"Failed (UTC): {datetime.now(timezone.utc).isoformat()}\n"
            f"Error: {exc}\n"
        )
        _notify(subject, body, ok=False)
        raise

@app.local_entrypoint()
def local_test():
    run_weekday_workflow.remote()
