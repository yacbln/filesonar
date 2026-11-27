"""Google Drive access: sign-in (OAuth for installed apps) and the few REST calls we use.

Everything goes through one ``httpx`` client. Requests are made only to Google's API and
sign-in hosts, plus the short-lived thumbnail links the Drive API hands out (checked to
be on ``googleusercontent.com`` before they are fetched). Downloads are streamed with a
size limit, so a large file can never be pulled into memory by accident.

Sign-in uses the loopback flow with PKCE and a random ``state``: the browser is sent to
Google, and Google redirects back to a one-shot server on 127.0.0.1. The resulting
refresh token is stored in a 0600 file in the data folder and never logged.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import httpx

log = logging.getLogger(__name__)

API = "https://www.googleapis.com/drive/v3"
UPLOAD_API = "https://www.googleapis.com/upload/drive/v3"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105 - an endpoint, not a secret
REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# drive.readonly: read and search every file (for indexing).
# drive.file: create files and folders, and change only those (for share_file).
SCOPES = (
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
)
FOLDER_MIME = "application/vnd.google-apps.folder"

FILE_FIELDS = (
    "id,name,mimeType,size,md5Checksum,modifiedTime,parents,webViewLink,trashed,"
    "ownedByMe,thumbnailLink,imageMediaMetadata(time,cameraMake,cameraModel),"
    "videoMediaMetadata(durationMillis,width,height)"
)
_RETRY_STATUS = {429, 500, 502, 503, 504}


class DriveError(RuntimeError):
    pass


class NotSignedIn(DriveError):
    def __init__(self) -> None:
        super().__init__("not signed in to Google Drive; run `filesonar drive login`")


# -- credentials -------------------------------------------------------------


@dataclass
class Token:
    client_id: str
    client_secret: str
    refresh_token: str
    account: str | None = None
    access_token: str | None = None
    expires_at: float = 0.0


def token_path(data_dir: Path) -> Path:
    return data_dir / "drive-token.json"


def save_token(path: Path, token: Token) -> None:
    """Write the token file atomically with 0600 permissions."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "client_id": token.client_id,
        "client_secret": token.client_secret,
        "refresh_token": token.refresh_token,
        "account": token.account,
    }
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as fh:
        os.fchmod(fh.fileno(), 0o600)
        json.dump(payload, fh)
    os.replace(tmp, path)


def load_token(path: Path) -> Token:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise NotSignedIn() from None
    if st.st_mode & 0o077:
        raise DriveError(f"{path} is readable by other users; run: chmod 600 {path}")
    raw = json.loads(path.read_text())
    try:
        return Token(
            client_id=raw["client_id"],
            client_secret=raw["client_secret"],
            refresh_token=raw["refresh_token"],
            account=raw.get("account"),
        )
    except KeyError:
        raise DriveError(f"{path} is incomplete; run `filesonar drive login` again") from None


def read_client_secret(path: Path) -> tuple[str, str]:
    """Client id and secret from the JSON downloaded from Google Cloud Console."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DriveError(f"cannot read {path}: {exc}") from None
    section = raw.get("installed")
    if not isinstance(section, dict):
        raise DriveError(
            f"{path} is not a desktop-app OAuth client. In Google Cloud Console create "
            "an OAuth client ID of type 'Desktop app' and download its JSON."
        )
    try:
        return section["client_id"], section["client_secret"]
    except KeyError:
        raise DriveError(f"{path} has no client_id/client_secret") from None


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge.decode().rstrip("=")


class _Redirect(http.server.BaseHTTPRequestHandler):
    result: dict[str, str] = {}  # noqa: RUF012 - replaced per server instance
    expected_path = "/"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != self.expected_path:
            self.send_error(404)
            return
        params = dict(urllib.parse.parse_qsl(parsed.query))
        if "code" not in params and "error" not in params:
            self.send_error(400)
            return
        type(self).result.update(params)
        body = (
            b"<html><body style='font-family:sans-serif'><h2>filesonar</h2>"
            b"<p>You can close this tab and return to the terminal.</p></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:  # keep the code out of the terminal log
        pass


def login(
    client_id: str,
    client_secret: str,
    *,
    open_browser: bool = True,
    show: Callable[[str], None] = print,
    timeout: float = 300.0,
    http_client: httpx.Client | None = None,
) -> Token:
    """Run the browser sign-in and return a token with a refresh token."""
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(24)
    path = "/" + secrets.token_urlsafe(8)

    handler = type("Handler", (_Redirect,), {"result": {}, "expected_path": path})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    redirect_uri = f"http://127.0.0.1:{server.server_port}{path}"
    url = (
        AUTH_URL
        + "?"
        + urllib.parse.urlencode(
            {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": " ".join(SCOPES),
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": state,
                "access_type": "offline",
                "prompt": "consent",
            }
        )
    )
    show(f"Open this link to sign in to Google Drive:\n\n  {url}\n")
    if open_browser:
        webbrowser.open(url)

    deadline = time.monotonic() + timeout
    server.timeout = 1.0
    try:
        while not handler.result and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    result = handler.result
    if not result:
        raise DriveError("sign-in timed out")
    if "error" in result:
        raise DriveError(f"sign-in was not completed: {result['error']}")
    if not secrets.compare_digest(result.get("state", ""), state):
        raise DriveError("sign-in response did not match this request; try again")

    client = http_client or httpx.Client(timeout=30)
    resp = client.post(
        TOKEN_URL,
        data={
            "code": result["code"],
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": verifier,
        },
    )
    if resp.status_code != 200:
        raise DriveError(f"Google refused the sign-in ({resp.status_code}): {_reason(resp)}")
    data = resp.json()
    granted = set(data.get("scope", "").split())
    missing = [s.rsplit("/", 1)[-1] for s in SCOPES if s not in granted]
    if missing:
        raise DriveError(
            "sign-in did not grant: " + ", ".join(missing) + ". Tick every box on "
            "Google's consent screen and try again."
        )
    if "refresh_token" not in data:
        raise DriveError("Google did not return a refresh token; try again")
    return Token(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=data["refresh_token"],
        access_token=data["access_token"],
        expires_at=time.time() + float(data.get("expires_in", 3600)),
    )


def _reason(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:200]
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("message", err))[:300]
    return str(body.get("error_description") or err or body)[:300]


# -- API client --------------------------------------------------------------


def is_thumbnail_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and (
        host == "googleusercontent.com" or host.endswith(".googleusercontent.com")
    )


class DriveClient:
    def __init__(
        self,
        token: Token,
        *,
        on_token: Callable[[Token], None] | None = None,
        http: httpx.Client | None = None,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.token = token
        self.on_token = on_token
        self.http = http or httpx.Client(timeout=timeout, follow_redirects=False)
        self.sleep = sleep
        self._lock = threading.Lock()

    @classmethod
    def from_data_dir(cls, data_dir: Path, **kwargs) -> DriveClient:
        path = token_path(data_dir)
        token = load_token(path)

        def store(updated: Token) -> None:
            save_token(path, updated)

        return cls(token, on_token=store, **kwargs)

    def close(self) -> None:
        self.http.close()

    # -- auth

    def _refresh(self) -> None:
        resp = self.http.post(
            TOKEN_URL,
            data={
                "client_id": self.token.client_id,
                "client_secret": self.token.client_secret,
                "refresh_token": self.token.refresh_token,
                "grant_type": "refresh_token",
            },
        )
        if resp.status_code != 200:
            if resp.status_code in (400, 401):
                raise DriveError(
                    "Google Drive access was revoked or has expired; run "
                    "`filesonar drive login` again"
                )
            raise DriveError(f"token refresh failed ({resp.status_code}): {_reason(resp)}")
        data = resp.json()
        self.token.access_token = data["access_token"]
        self.token.expires_at = time.time() + float(data.get("expires_in", 3600))
        if data.get("refresh_token") and data["refresh_token"] != self.token.refresh_token:
            self.token.refresh_token = data["refresh_token"]
            if self.on_token:
                self.on_token(self.token)

    def _auth_header(self) -> dict[str, str]:
        with self._lock:
            if not self.token.access_token or time.time() > self.token.expires_at - 60:
                self._refresh()
            return {"Authorization": f"Bearer {self.token.access_token}"}

    # -- requests

    def _send(
        self,
        method: str,
        url: str,
        *,
        stream: bool = False,
        body: Callable[[], Iterator[bytes]] | None = None,
        **kwargs,
    ) -> httpx.Response:
        """Send with auth, one re-auth on 401 and backoff on rate limits/server errors.

        ``body`` makes a fresh request body for each attempt (a file streamed from
        disk can only be read once per attempt)."""
        delay = 1.0
        reauthed = False
        for attempt in range(6):
            headers = {**kwargs.pop("headers", {}), **self._auth_header()}
            extra = {"content": body()} if body is not None else {}
            request = self.http.build_request(method, url, headers=headers, **kwargs, **extra)
            kwargs["headers"] = {k: v for k, v in headers.items() if k != "Authorization"}
            resp = self.http.send(request, stream=stream)
            if resp.status_code == 401 and not reauthed:
                resp.close()
                with self._lock:
                    self.token.access_token = None
                reauthed = True
                continue
            if resp.status_code in _RETRY_STATUS or _is_rate_limited(resp):
                resp.close()
                if attempt == 5:
                    break
                self.sleep(delay)
                delay = min(delay * 2, 32)
                continue
            return resp
        raise DriveError(f"Google Drive is not responding ({method} {url.split('?')[0]})")

    def _json(self, method: str, url: str, **kwargs) -> dict:
        resp = self._send(method, url, **kwargs)
        if resp.status_code >= 400:
            raise DriveError(f"Drive API error {resp.status_code}: {_reason(resp)}")
        return resp.json()

    def _download(
        self, url: str, max_bytes: int, *, auth: bool = True, sink: BinaryIO | None = None, **kwargs
    ) -> bytes:
        """The response body (at most ``max_bytes``), or written to ``sink`` if given."""
        if auth:
            resp = self._send("GET", url, stream=True, **kwargs)
        else:
            resp = self.http.send(self.http.build_request("GET", url, **kwargs), stream=True)
        try:
            if resp.status_code >= 300:  # redirects are not followed: treat as failure
                resp.read()
                raise DriveError(f"download failed ({resp.status_code}): {_reason(resp)}")
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise DriveError("file is larger than the size limit")
            chunks, total = [], 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise DriveError("file is larger than the size limit")
                if sink is not None:
                    sink.write(chunk)
                else:
                    chunks.append(chunk)
            return b"".join(chunks)
        finally:
            resp.close()

    # -- Drive calls

    def account(self) -> str:
        data = self._json("GET", f"{API}/about", params={"fields": "user(emailAddress)"})
        return data.get("user", {}).get("emailAddress", "")

    def start_page_token(self) -> str:
        return self._json("GET", f"{API}/changes/startPageToken")["startPageToken"]

    def list_files(self, query: str, fields: str = FILE_FIELDS) -> Iterator[dict]:
        page: str | None = None
        while True:
            params = {
                "q": query,
                "fields": f"nextPageToken,files({fields})",
                "pageSize": 1000,
                "spaces": "drive",
            }
            if page:
                params["pageToken"] = page
            data = self._json("GET", f"{API}/files", params=params)
            yield from data.get("files", [])
            page = data.get("nextPageToken")
            if not page:
                return

    def changes(self, page_token: str) -> tuple[list[dict], str]:
        """All changes since ``page_token`` and the token to use next time."""
        changes: list[dict] = []
        token = page_token
        while True:
            data = self._json(
                "GET",
                f"{API}/changes",
                params={
                    "pageToken": token,
                    "fields": f"nextPageToken,newStartPageToken,"
                    f"changes(fileId,removed,file({FILE_FIELDS}))",
                    "pageSize": 1000,
                    "spaces": "drive",
                    "includeRemoved": "true",
                },
            )
            changes.extend(data.get("changes", []))
            if "newStartPageToken" in data:
                return changes, data["newStartPageToken"]
            token = data["nextPageToken"]

    def get_file(self, file_id: str) -> dict:
        return self._json("GET", f"{API}/files/{_quote(file_id)}", params={"fields": FILE_FIELDS})

    def download(self, file_id: str, max_bytes: int, sink: BinaryIO | None = None) -> bytes:
        return self._download(
            f"{API}/files/{_quote(file_id)}", max_bytes, params={"alt": "media"}, sink=sink
        )

    def export(
        self, file_id: str, mime: str, max_bytes: int, sink: BinaryIO | None = None
    ) -> bytes:
        return self._download(
            f"{API}/files/{_quote(file_id)}/export",
            max_bytes,
            params={"mimeType": mime},
            sink=sink,
        )

    def thumbnail(self, link: str, max_side: int, max_bytes: int) -> bytes:
        if not is_thumbnail_url(link):
            raise DriveError("unexpected thumbnail address")
        # Links end in "=s220"; ask for a larger rendition of the same image.
        base, sep, size = link.rpartition("=s")
        url = f"{base}=s{max_side}" if sep and size.isdigit() else link
        return self._download(url, max_bytes)

    def find_folder(self, name: str) -> str | None:
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        query = f"name = '{escaped}' and mimeType = '{FOLDER_MIME}' and trashed = false"
        for item in self.list_files(query, fields="id,name"):
            return item["id"]
        return None

    def create_folder(self, name: str) -> str:
        return self._json(
            "POST",
            f"{API}/files",
            params={"fields": "id"},
            json={"name": name, "mimeType": FOLDER_MIME},
        )["id"]

    def upload(self, name: str, data: bytes | int, mime: str, parent: str, size: int = -1) -> dict:
        """Upload a new private file in ``parent`` (resumable protocol).

        ``data`` is the content, or an open file descriptor to stream from (with
        ``size``), so large videos are never held in memory."""
        if isinstance(data, bytes):
            size = len(data)

            def body() -> Iterator[bytes]:
                yield data
        else:
            fd = data

            def body() -> Iterator[bytes]:
                offset = 0
                while offset < size:
                    chunk = os.pread(fd, min(8 * 1024 * 1024, size - offset), offset)
                    if not chunk:
                        raise DriveError("the file got shorter while it was being uploaded")
                    offset += len(chunk)
                    yield chunk

        start = self._send(
            "POST",
            f"{UPLOAD_API}/files",
            params={"uploadType": "resumable", "fields": FILE_FIELDS},
            json={"name": name, "parents": [parent]},
            headers={"X-Upload-Content-Type": mime, "X-Upload-Content-Length": str(size)},
        )
        if start.status_code >= 400:
            raise DriveError(f"upload refused ({start.status_code}): {_reason(start)}")
        location = start.headers.get("Location", "")
        if not location.startswith(UPLOAD_API + "/"):
            raise DriveError("unexpected upload address")
        resp = self._send(
            "PUT",
            location,
            body=body,
            headers={"Content-Type": mime, "Content-Length": str(size)},
        )
        if resp.status_code >= 400:
            raise DriveError(f"upload failed ({resp.status_code}): {_reason(resp)}")
        return resp.json()

    def revoke(self) -> None:
        self.http.post(REVOKE_URL, data={"token": self.token.refresh_token})


def _is_rate_limited(resp: httpx.Response) -> bool:
    if resp.status_code != 403:
        return False
    try:
        errors = resp.json().get("error", {}).get("errors", [])
    except ValueError:
        return False
    return any(e.get("reason") in ("rateLimitExceeded", "userRateLimitExceeded") for e in errors)


def _quote(file_id: str) -> str:
    return urllib.parse.quote(file_id, safe="")


def sign_in(data_dir: Path, client_secret_file: Path | None = None, *, open_browser=True) -> Token:
    """Sign in and save the token. Without a client_secret file, reuse the OAuth client
    saved with the last sign-in (people are told they may delete the downloaded file)."""
    path = token_path(data_dir)
    if client_secret_file is not None:
        client_id, client_secret = read_client_secret(client_secret_file)
    else:
        try:
            saved = load_token(path)
        except NotSignedIn:
            raise DriveError(
                "the first sign-in needs --client-secret, the client_secret_*.json "
                "downloaded from Google Cloud Console"
            ) from None
        client_id, client_secret = saved.client_id, saved.client_secret
    token = login(client_id, client_secret, open_browser=open_browser)
    client = DriveClient(token)
    try:
        token.account = client.account()
    finally:
        client.close()
    save_token(path, token)
    return token
