"""Load the same read-only reporting identity from a file or a Modal secret."""
import json
from pathlib import Path


def reporting_credentials(value: str, scopes: list[str]):
    from google.oauth2 import service_account

    if value.lstrip().startswith("{"):
        try:
            info = json.loads(value)
            return service_account.Credentials.from_service_account_info(info, scopes=scopes)
        except Exception:
            # Never include inline secret material in an exception or a run log.
            raise ValueError("Google reporting credential JSON is invalid") from None
    path = Path(value)
    if not value or not path.is_file():
        raise FileNotFoundError(
            "Google reporting credential is unavailable in this runtime; configure "
            "GOOGLE_SERVICE_ACCOUNT_JSON as a secret or mount the configured key file"
        )
    return service_account.Credentials.from_service_account_file(str(path), scopes=scopes)
