"""An in-memory Google Drive API, served through httpx.MockTransport.

It implements just the endpoints filesonar uses, closely enough to exercise paging,
change tokens, exports, previews, uploads, token refresh and error handling.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from dataclasses import dataclass, field

import httpx

from filesonar.drive import FOLDER_MIME, DriveClient, Token

ROOT = "root0"


@dataclass
class FakeFile:
    id: str
    name: str
    mime: str
    parents: list[str]
    content: bytes = b""
    trashed: bool = False
    owned: bool = True
    modified: str = "2024-05-01T10:00:00.000Z"
    export: dict[str, bytes] = field(default_factory=dict)
    photo: dict | None = None
    thumbnail: bytes | None = None

    def meta(self) -> dict:
        out = {
            "id": self.id,
            "name": self.name,
            "mimeType": self.mime,
            "parents": self.parents,
            "trashed": self.trashed,
            "ownedByMe": self.owned,
            "modifiedTime": self.modified,
            "webViewLink": f"https://drive.google.com/file/d/{self.id}/view",
        }
        if not self.mime.startswith("application/vnd.google-apps"):
            out["size"] = str(len(self.content))
            out["md5Checksum"] = hashlib.md5(self.content).hexdigest()
        if self.photo:
            out["imageMediaMetadata"] = self.photo
        if self.thumbnail is not None:
            out["thumbnailLink"] = f"https://lh3.googleusercontent.com/t/{self.id}=s220"
        return out


class FakeDrive:
    def __init__(self) -> None:
        self.files: dict[str, FakeFile] = {}
        self.changes: list[str] = []  # file ids, in order; the page token is an index
        self.calls: list[tuple[str, str]] = []
        self.downloads: list[str] = []
        self.uploads: list[str] = []
        self.thumbnails: list[str] = []
        self.access_tokens = 0
        self.fail_next: list[int] = []  # status codes to return before answering
        self.broken: set[str] = set()  # file ids whose content cannot be downloaded
        self.revoked = False
        self.expire_access = False
        self._next_id = 0
        self._sessions: dict[str, dict] = {}

    # -- test helpers

    def add(self, name: str, content: bytes = b"", mime: str = "", parent: str = ROOT, **kw):
        self._next_id += 1
        file_id = kw.pop("id", f"f{self._next_id}")
        if not mime:
            mime = _guess(name)
        f = FakeFile(file_id, name, mime, [parent], content, **kw)
        self.files[file_id] = f
        self.changes.append(file_id)
        return f

    def folder(self, name: str, parent: str = ROOT, **kw) -> FakeFile:
        return self.add(name, mime=FOLDER_MIME, parent=parent, **kw)

    def touch(self, file_id: str, **changes) -> None:
        f = self.files[file_id]
        for key, value in changes.items():
            setattr(f, key, value)
        self.changes.append(file_id)

    def delete(self, file_id: str) -> None:
        del self.files[file_id]
        self.changes.append(file_id)

    def client(self, **kwargs) -> DriveClient:
        token = Token("cid", "csecret", "refresh-1", account="me@example.com")
        http = httpx.Client(transport=httpx.MockTransport(self.handle))
        return DriveClient(token, http=http, sleep=lambda _s: None, **kwargs)

    # -- server

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        self.calls.append((request.method, url.path))
        if url.host == "oauth2.googleapis.com":
            return self._token(request)
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0), json={"error": {"message": "busy"}})
        auth = request.headers.get("Authorization", "")
        if url.host == "lh3.googleusercontent.com":
            return self._thumbnail(url)
        if not auth.startswith("Bearer access-") or (
            self.expire_access and auth != f"Bearer access-{self.access_tokens}"
        ):
            return httpx.Response(401, json={"error": {"message": "invalid credentials"}})
        params = dict(url.params)
        path = url.path
        if path == "/drive/v3/about":
            return httpx.Response(200, json={"user": {"emailAddress": "me@example.com"}})
        if path == "/drive/v3/changes/startPageToken":
            return httpx.Response(200, json={"startPageToken": str(len(self.changes))})
        if path == "/drive/v3/changes":
            return self._changes(int(params["pageToken"]))
        if path == "/drive/v3/files" and request.method == "GET":
            return self._list(params)
        if path == "/drive/v3/files" and request.method == "POST":
            body = json.loads(request.content)
            f = self.add(body["name"], mime=body["mimeType"])
            return httpx.Response(200, json={"id": f.id})
        if path == "/upload/drive/v3/files" and request.method == "POST":
            session = f"s{len(self._sessions)}"
            self._sessions[session] = json.loads(request.content)
            loc = f"https://www.googleapis.com/upload/drive/v3/files?upload_id={session}"
            return httpx.Response(200, headers={"Location": loc})
        if path == "/upload/drive/v3/files" and request.method == "PUT":
            body = self._sessions.pop(params["upload_id"])
            f = self.add(body["name"], request.content, parent=body["parents"][0])
            self.uploads.append(f.id)
            return httpx.Response(200, json=f.meta())
        m = re.fullmatch(r"/drive/v3/files/([^/]+)(/export)?", path)
        if m:
            file_id = urllib.parse.unquote(m.group(1))
            if file_id == "root":
                return httpx.Response(200, json={"id": ROOT, "name": "My Drive"})
            f = self.files.get(file_id)
            if f is None:
                return httpx.Response(404, json={"error": {"message": "File not found"}})
            if m.group(2):
                if params["mimeType"] not in f.export:
                    # What Drive says when a file is too large to export in a format.
                    return httpx.Response(
                        403,
                        json={"error": {"message": "This file is too large to be exported.",
                                        "errors": [{"reason": "exportSizeLimitExceeded"}]}},
                    )  # fmt: skip
                self.downloads.append(file_id)
                return httpx.Response(200, content=f.export[params["mimeType"]])
            if params.get("alt") == "media":
                if file_id in self.broken:
                    return httpx.Response(500, json={"error": {"message": "backend error"}})
                self.downloads.append(file_id)
                return httpx.Response(200, content=f.content)
            return httpx.Response(200, json=f.meta())
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = dict(urllib.parse.parse_qsl(request.content.decode()))
        if url_path(request) == "/revoke":
            self.revoked = True
            return httpx.Response(200)
        if self.revoked or form.get("refresh_token") not in (None, "refresh-1"):
            return httpx.Response(400, json={"error": "invalid_grant"})
        self.access_tokens += 1
        return httpx.Response(
            200, json={"access_token": f"access-{self.access_tokens}", "expires_in": 3600}
        )

    def _thumbnail(self, url: httpx.URL) -> httpx.Response:
        file_id = url.path.split("/")[-1].split("=")[0]
        self.thumbnails.append(file_id)
        f = self.files.get(file_id)
        if f is None or f.thumbnail is None:
            return httpx.Response(404)
        return httpx.Response(200, content=f.thumbnail)

    def _matches(self, f: FakeFile, q: str) -> bool:
        if "trashed = false" in q and f.trashed:
            return False
        if f"mimeType = '{FOLDER_MIME}'" in q and f.mime != FOLDER_MIME:
            return False
        if f"mimeType != '{FOLDER_MIME}'" in q and f.mime == FOLDER_MIME:
            return False
        if "'me' in owners" in q and not f.owned:
            return False
        parent = re.search(r"'([^']+)' in parents", q)
        if parent and parent.group(1) not in f.parents:
            return False
        m = re.search(r"name = '((?:[^'\\]|\\.)*)'", q)
        return not (m and f.name != m.group(1).replace("\\'", "'").replace("\\\\", "\\"))

    def _list(self, params: dict) -> httpx.Response:
        found = [f.meta() for f in self.files.values() if self._matches(f, params["q"])]
        start = int(params.get("pageToken", 0))
        size = 2  # small pages, so paging is exercised
        body: dict = {"files": found[start : start + size]}
        if start + size < len(found):
            body["nextPageToken"] = str(start + size)
        return httpx.Response(200, json=body)

    def _changes(self, start: int) -> httpx.Response:
        ids = self.changes[start : start + 2]
        out = []
        for file_id in ids:
            f = self.files.get(file_id)
            if f is None:
                out.append({"fileId": file_id, "removed": True})
            else:
                out.append({"fileId": file_id, "removed": False, "file": f.meta()})
        body: dict = {"changes": out}
        if start + 2 < len(self.changes):
            body["nextPageToken"] = str(start + 2)
        else:
            body["newStartPageToken"] = str(len(self.changes))
        return httpx.Response(200, json=body)


def url_path(request: httpx.Request) -> str:
    return request.url.path


def _guess(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower()
    return {
        "pdf": "application/pdf",
        "txt": "text/plain",
        "jpg": "image/jpeg",
        "png": "image/png",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "zip": "application/zip",
    }.get(ext, "application/octet-stream")
