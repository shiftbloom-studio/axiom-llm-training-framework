"""Public http(s) fetch guard for the optional UI harvest.

Operators may name web documents to download. Only http and https URLs whose
names resolve entirely to public addresses are fetched. Redirects are refused
until the next target passes the same check, so a public page cannot forward
the client at loopback, link-local metadata, or another non-public host.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.parse
import urllib.request
from typing import override

_MAX_REDIRECTS = 5
_PUBLIC_HTTP_URL = re.compile(r"https?://\S+", re.IGNORECASE)


class PublicUrlError(ValueError):
    """The URL is not a public http(s) target that harvest may fetch."""


class _RedirectError(Exception):
    def __init__(self, target: str) -> None:
        super().__init__(target)
        self.target = target


class _StopRedirect(urllib.request.HTTPRedirectHandler):
    """Surface the next Location instead of letting urllib open it."""

    @override
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            fp.read()
        finally:
            fp.close()
        raise _RedirectError(str(newurl))


def validate_public_http_url(url: str) -> str:
    """Return `url` when it is an http(s) URL that resolves only to public addresses."""
    if not isinstance(url, str) or not url:
        raise PublicUrlError("URL must be a non-empty string")
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise PublicUrlError("URL could not be parsed") from exc
    if parts.scheme.lower() not in {"http", "https"}:
        raise PublicUrlError("only http and https URLs are allowed")
    if parts.username is not None or parts.password is not None:
        raise PublicUrlError("URLs with credentials are not allowed")
    host = parts.hostname
    if not host:
        raise PublicUrlError("URL must include a host")
    for address in _resolve_host(host):
        if _is_blocked_address(address):
            raise PublicUrlError("host resolves to a non-public address")
    return url


def read_public_http_text(url: str, *, timeout: float = 12, max_chars: int = 200_000) -> str:
    """Fetch `url` as text, following only redirects that stay on public http(s)."""
    current = validate_public_http_url(url)
    opener = urllib.request.build_opener(_StopRedirect)
    for _ in range(_MAX_REDIRECTS + 1):
        # re.fullmatch is the barrier CodeQL's py/full-ssrf query treats as sanitizing
        # the URL that reaches the client. The address check above is what makes that true.
        if _PUBLIC_HTTP_URL.fullmatch(current):
            try:
                with opener.open(urllib.request.Request(current), timeout=timeout) as response:
                    return response.read().decode("utf-8", "ignore")[:max_chars]
            except _RedirectError as redirect:
                current = validate_public_http_url(urllib.parse.urljoin(current, redirect.target))
                continue
        raise PublicUrlError("URL is not an allowed http(s) URL")
    raise PublicUrlError("too many redirects")


def _resolve_host(host: str) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    literal = _literal_address(host)
    if literal is not None:
        return (literal,)
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise PublicUrlError("host could not be resolved") from exc
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    seen: set[str] = set()
    for *_, sockaddr in infos:
        raw = str(sockaddr[0]).split("%", 1)[0]
        if raw in seen:
            continue
        seen.add(raw)
        addresses.append(ipaddress.ip_address(raw))
    if not addresses:
        raise PublicUrlError("host could not be resolved")
    return tuple(addresses)


def _literal_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address):
        mapped = address.ipv4_mapped
        if mapped is not None and _is_blocked_address(mapped):
            return True
    return (
        address.is_multicast
        or address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
        or not address.is_global
    )
