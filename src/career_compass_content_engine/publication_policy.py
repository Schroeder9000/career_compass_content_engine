"""Fail-closed publication decisions, independent of model and network calls."""
from typing import Any

MAX_REVISIONS = 3


def metadata_blockers(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict):
        return ["Publishing metadata must be a JSON object."]
    issues = []
    if metadata.get("content_ready") is not True:
        issues.append("Publishing QA did not explicitly approve the article.")
    blockers = metadata.get("blocking_issues")
    if not isinstance(blockers, list) or any(not isinstance(x, str) for x in blockers):
        issues.append("Publishing QA did not return a valid blocking_issues list.")
    elif blockers:
        issues.extend(blockers)
    for field in ("title", "summary"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            issues.append(f"Publishing metadata is missing {field}.")
    return issues


def route_after_evaluate(state: dict) -> str:
    if state.get("evaluation_passed") is True:
        return "node_prepare_metadata"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "node_blocked"
    return "node_edit"


def route_after_metadata(state: dict) -> str:
    return "node_blocked" if metadata_blockers(state.get("metadata_payload")) else "node_publish"
