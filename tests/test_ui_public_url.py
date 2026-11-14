"""Unit tests for the UI harvest public-URL guard."""

from __future__ import annotations

import importlib.util
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Self

import pytest

_PATH = Path(__file__).resolve().parents[1] / "ui" / "public_url.py"
_SPEC = importlib.util.spec_from_file_location("ui_public_url", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
public_url = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(public_url)

_PUBLIC_ADDR = "93.184.216.34"


def _dns(address: str):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

    return fake_getaddrinfo


def test_allows_public_http_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(public_url.socket, "getaddrinfo", _dns(_PUBLIC_ADDR))
    url = "https://example.com/papers/a?q=1"
    assert public_url.validate_public_http_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://127.0.0.1/",
        "http://10.1.2.3/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
    ],
)
def test_rejects_nonpublic_urls(url: str) -> None:
    with pytest.raises(public_url.PublicUrlError):
        public_url.validate_public_http_url(url)


def test_rejects_name_that_resolves_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(public_url.socket, "getaddrinfo", _dns("169.254.169.254"))
    with pytest.raises(public_url.PublicUrlError):
        public_url.validate_public_http_url("https://metadata.example/latest")


def test_rejects_when_any_resolved_address_is_private(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (_PUBLIC_ADDR, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0)),
        ]

    monkeypatch.setattr(public_url.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(public_url.PublicUrlError):
        public_url.validate_public_http_url("https://example.com/mixed")


def test_does_not_follow_redirect_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(public_url.socket, "getaddrinfo", _dns(_PUBLIC_ADDR))
    opened: list[str] = []

    class _Opener:
        def open(self, fullurl, data=None, timeout=None):
            opened.append(fullurl.full_url)
            raise public_url._RedirectError("http://169.254.169.254/latest/meta-data")

    monkeypatch.setattr(
        public_url.urllib.request,
        "build_opener",
        lambda *args, **kwargs: _Opener(),
    )
    with pytest.raises(public_url.PublicUrlError):
        public_url.read_public_http_text("https://example.com/start")
    assert opened == ["https://example.com/start"]


def test_follows_redirect_between_public_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(public_url.socket, "getaddrinfo", _dns(_PUBLIC_ADDR))
    opened: list[str] = []

    class _Body:
        def read(self) -> bytes:
            return b"hello"

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    class _Opener:
        def open(self, fullurl, data=None, timeout=None):
            opened.append(fullurl.full_url)
            if len(opened) == 1:
                raise public_url._RedirectError("https://cdn.example.com/file")
            return _Body()

    monkeypatch.setattr(
        public_url.urllib.request,
        "build_opener",
        lambda *args, **kwargs: _Opener(),
    )
    assert public_url.read_public_http_text("https://example.com/a") == "hello"
    assert opened == ["https://example.com/a", "https://cdn.example.com/file"]


def test_live_redirect_is_checked_before_it_is_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The urllib handler must not open a redirect target that fails validation."""
    real_blocked = public_url._is_blocked_address

    def allow_loopback(address: object) -> bool:
        if getattr(address, "is_loopback", False):
            return False
        return real_blocked(address)

    monkeypatch.setattr(public_url, "_is_blocked_address", allow_loopback)

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/ok":
                body = b"hello"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data")
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        assert public_url.read_public_http_text(f"http://127.0.0.1:{port}/ok") == "hello"
        with pytest.raises(public_url.PublicUrlError):
            public_url.read_public_http_text(f"http://127.0.0.1:{port}/go")
    finally:
        server.shutdown()
        server.server_close()
