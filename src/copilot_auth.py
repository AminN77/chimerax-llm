# vim: set expandtab shiftwidth=4 softtabstop=4:

"""GitHub Copilot OAuth authentication (device flow) and Copilot API token exchange.

Device flow yields a GitHub OAuth access token. The Copilot chat API
(``api.githubcopilot.com``) expects a **Copilot JWT** from
``GET https://api.github.com/copilot_internal/v2/token`` authenticated with that
OAuth token. JWTs expire (~25 minutes); we refresh by re-exchanging when needed.

Reads/writes opencode's auth store (``~/.local/share/opencode/auth.json``) so both
tools can share credentials.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.error

# Same client ID as VS Code / opencode-github-copilot (public Copilot OAuth app).
# Device-flow tokens from this app work with copilot_internal token exchange.
_CLIENT_ID = "Iv1.b507a08c87ecfe98"
_DEVICE_CODE_URL = "https://github.com/login/device/code"
_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
# Match Copilot client expectations for the internal token endpoint (see opencode-github-copilot auth.ts).
_COPILOT_EXCHANGE_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.26.7",
    "Editor-Version": "vscode/1.99.3",
    "Editor-Plugin-Version": "copilot-chat/0.26.7",
    "Copilot-Integration-Id": "vscode-chat",
}
_POLL_SAFETY_MARGIN = 3  # seconds
# Default JWT lifetime if API omits expires_at (seconds).
_DEFAULT_JWT_TTL = 25 * 60
# Refresh this many seconds before expiry.
_REFRESH_SKEW = 120


def _opencode_auth_path() -> Path:
    """Path to opencode's auth.json."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "opencode" / "auth.json"
    xdg = os.environ.get("XDG_DATA_HOME", "")
    if xdg:
        return Path(xdg) / "opencode" / "auth.json"
    if os.uname().sysname == "Darwin":
        return Path.home() / ".local" / "share" / "opencode" / "auth.json"
    return Path.home() / ".local" / "share" / "opencode" / "auth.json"


def _load_auth_file() -> Dict[str, Any]:
    p = _opencode_auth_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _github_copilot_entry(data: Dict[str, Any]) -> Dict[str, Any]:
    ent = data.get("github-copilot")
    return ent if isinstance(ent, dict) else {}


def _looks_like_jwt(s: str) -> bool:
    """Copilot/GitHub JWTs from exchange are typically base64url segments starting with ``eyJ``."""
    t = s.strip()
    return len(t) > 20 and t.startswith("eyJ")


def _oauth_token_from_entry(ent: Dict[str, Any]) -> Optional[str]:
    """GitHub OAuth access token for device-flow (not the Copilot JWT)."""
    o = ent.get("oauth_access") or ent.get("refresh")
    if isinstance(o, str) and o.strip():
        return o.strip()
    acc = ent.get("access")
    if isinstance(acc, str) and acc.strip() and not _looks_like_jwt(acc):
        return acc.strip()
    return None


def _write_auth_file(data: Dict[str, Any]) -> None:
    p = _opencode_auth_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    os.chmod(p, 0o600)


def _parse_expires_at(raw: Any) -> Optional[float]:
    """Return Unix timestamp when JWT expires, or None if unknown."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # Could be seconds or milliseconds.
        if raw > 1e12:
            return float(raw) / 1000.0
        return float(raw)
    if isinstance(raw, str):
        try:
            # ISO8601 from GitHub
            from datetime import datetime

            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw).timestamp()
        except Exception:
            return None
    return None


def exchange_oauth_for_copilot_jwt(oauth_access_token: str) -> Tuple[str, float]:
    """Exchange a GitHub OAuth access token for a Copilot API JWT.

    GitHub expects ``Authorization: Bearer <oauth>`` (not ``token``) and the same
    Copilot-style headers VS Code sends; otherwise ``copilot_internal`` may return 404.

    Returns ``(jwt, expires_at_unix)``. ``expires_at_unix`` may be estimated if
    the API omits expiry metadata.
    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {oauth_access_token}",
        **_COPILOT_EXCHANGE_HEADERS,
    }
    req = urllib.request.Request(_COPILOT_TOKEN_URL, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Copilot token exchange failed ({e.code}): {body}") from e

    jwt = data.get("token")
    if not jwt or not isinstance(jwt, str):
        raise RuntimeError("Copilot token exchange: missing 'token' in response")

    exp = _parse_expires_at(data.get("expires_at"))
    if exp is None:
        exp = time.time() + _DEFAULT_JWT_TTL
    return jwt, float(exp)


def _save_copilot_auth(
    oauth_access: str,
    copilot_jwt: str,
    copilot_expires_at: float,
) -> None:
    """Persist OAuth token and Copilot JWT (with expiry) to the shared auth file."""
    existing = _load_auth_file()
    ent = _github_copilot_entry(existing)
    ent["type"] = "oauth"
    ent["oauth_access"] = oauth_access
    ent["copilot_jwt"] = copilot_jwt
    ent["copilot_expires_at"] = copilot_expires_at
    ent.pop("copilot_oauth_fallback", None)
    # Align with opencode: refresh = GitHub OAuth, access = Copilot API bearer (JWT).
    ent["refresh"] = oauth_access
    ent["access"] = copilot_jwt
    ent["expires"] = int(copilot_expires_at)
    existing["github-copilot"] = ent
    _write_auth_file(existing)


def get_oauth_access_token() -> Optional[str]:
    """Return stored GitHub OAuth access token if present."""
    return _oauth_token_from_entry(_github_copilot_entry(_load_auth_file()))


def get_token() -> Optional[str]:
    """Return a token suitable for *logged-in* checks (OAuth or legacy single field).

    Prefer OAuth access token when present so callers can exchange. For backward
    compatibility, falls back to ``copilot_jwt`` only if no OAuth is stored.
    """
    ent = _github_copilot_entry(_load_auth_file())
    o = _oauth_token_from_entry(ent)
    if o:
        return o
    jwt = ent.get("copilot_jwt")
    if isinstance(jwt, str) and jwt.strip():
        return jwt.strip()
    acc = ent.get("access")
    if isinstance(acc, str) and acc.strip():
        return acc.strip()
    return None


def _save_oauth_fallback_bearer(oauth: str) -> None:
    """Persist OAuth as API bearer when ``copilot_internal`` exchange is unavailable."""
    existing = _load_auth_file()
    ent = _github_copilot_entry(existing)
    exp = time.time() + _DEFAULT_JWT_TTL
    ent["type"] = "oauth"
    ent["oauth_access"] = oauth
    ent["copilot_jwt"] = oauth
    ent["copilot_expires_at"] = exp
    ent["copilot_oauth_fallback"] = True
    ent["refresh"] = oauth
    ent["access"] = oauth
    ent["expires"] = int(exp)
    existing["github-copilot"] = ent
    _write_auth_file(existing)


def get_copilot_token() -> Optional[str]:
    """Return a bearer token for ``api.githubcopilot.com``.

    Prefer a Copilot JWT from ``copilot_internal`` exchange. If exchange fails
    (e.g. wrong OAuth app), fall back to the GitHub OAuth access token, which some
    Copilot gateways still accept (same as pre-exchange ChimeraLLM behavior).
    """
    data = _load_auth_file()
    ent = _github_copilot_entry(data)
    now = time.time()

    oauth = _oauth_token_from_entry(ent)

    jwt = ent.get("copilot_jwt")
    if isinstance(jwt, str):
        jwt = jwt.strip() or None
    else:
        jwt = None

    exp_raw = ent.get("copilot_expires_at")
    exp: Optional[float] = None
    if isinstance(exp_raw, (int, float)):
        exp = float(exp_raw)
    elif exp_raw is not None:
        exp = _parse_expires_at(exp_raw)

    if jwt and exp is not None and now < exp - _REFRESH_SKEW:
        return jwt

    if oauth:
        try:
            new_jwt, new_exp = exchange_oauth_for_copilot_jwt(oauth)
            _save_copilot_auth(oauth, new_jwt, new_exp)
            return new_jwt
        except Exception:
            _save_oauth_fallback_bearer(oauth)
            return oauth

    # Legacy: only JWT stored, try using it (may 401 until user re-logs in)
    if jwt:
        return jwt

    acc = ent.get("access")
    if isinstance(acc, str) and acc.strip() and _looks_like_jwt(acc.strip()):
        return acc.strip()

    return None


# ---------------------------------------------------------------------------
# Device-flow login (interactive)
# ---------------------------------------------------------------------------


class DeviceFlowError(RuntimeError):
    pass


def _post_json(url: str, body: dict) -> dict:
    """POST JSON and return parsed response."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def start_device_flow() -> dict:
    """Initiate a device-flow login.  Returns a dict with keys:

    - ``verification_uri``  – URL the user should open
    - ``user_code``         – code to enter on that page
    - ``device_code``       – opaque code for polling  (internal)
    - ``interval``          – poll interval in seconds  (internal)
    """
    return _post_json(
        _DEVICE_CODE_URL,
        {
            "client_id": _CLIENT_ID,
            "scope": "read:user",
        },
    )


def poll_for_token(device_code: str, interval: int = 5, timeout: int = 300) -> str:
    """Poll GitHub until the user completes the device-flow authorization.

    Returns the **Copilot API JWT** (after exchange). Raises ``DeviceFlowError``
    on failure or timeout.
    """
    deadline = time.monotonic() + timeout
    poll_interval = interval

    oauth_token: Optional[str] = None
    while time.monotonic() < deadline:
        time.sleep(poll_interval + _POLL_SAFETY_MARGIN)

        try:
            data = _post_json(
                _ACCESS_TOKEN_URL,
                {
                    "client_id": _CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
        except urllib.error.HTTPError:
            raise DeviceFlowError("HTTP error while polling for token")

        if data.get("access_token"):
            oauth_token = data["access_token"]
            break

        error = data.get("error", "")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            poll_interval = data.get("interval", poll_interval + 5)
            continue
        if error:
            raise DeviceFlowError(f"Device flow error: {error}")

    if not oauth_token:
        raise DeviceFlowError("Timed out waiting for authorization")

    try:
        jwt, exp = exchange_oauth_for_copilot_jwt(oauth_token)
        _save_copilot_auth(oauth_token, jwt, exp)
        return jwt
    except Exception:
        _save_oauth_fallback_bearer(oauth_token)
        return oauth_token
