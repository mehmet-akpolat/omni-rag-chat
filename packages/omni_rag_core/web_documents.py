import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlsplit

import httpx

from .documents import InvalidDocument


@dataclass(frozen=True)
class WebPage:
    url: str
    title: str
    text: str


class _ReadableHtmlParser(HTMLParser):
    _ignored_tags = {
        "aside",
        "canvas",
        "footer",
        "form",
        "iframe",
        "nav",
        "noscript",
        "script",
        "style",
        "svg",
    }
    _block_tags = {"blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "pre", "td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.body_text: list[str] = []
        self.blocks: list[str] = []
        self.preferred_blocks: list[str] = []
        self._ignored_depth = 0
        self._body_depth = 0
        self._preferred_depth = 0
        self._in_title = False
        self._block_tag: str | None = None
        self._block_parts: list[str] = []
        self._block_is_preferred = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if self._ignored_depth:
            if tag in self._ignored_tags:
                self._ignored_depth += 1
            return
        if tag in self._ignored_tags:
            self._ignored_depth = 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "body":
            self._body_depth += 1
        if tag in {"article", "main"}:
            self._preferred_depth += 1
        if tag in self._block_tags and self._block_tag is None:
            self._block_tag = tag
            self._block_parts = []
            self._block_is_preferred = self._preferred_depth > 0
        elif tag == "br" and self._block_tag:
            self._block_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._ignored_depth:
            if tag in self._ignored_tags:
                self._ignored_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag == self._block_tag:
            text = " ".join("".join(self._block_parts).split())
            if text:
                if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
                    text = f"{'#' * int(tag[1])} {text}"
                self.blocks.append(text)
                if self._block_is_preferred:
                    self.preferred_blocks.append(text)
            self._block_tag = None
            self._block_parts = []
            self._block_is_preferred = False
        if tag in {"article", "main"} and self._preferred_depth:
            self._preferred_depth -= 1
        if tag == "body" and self._body_depth:
            self._body_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        if self._body_depth:
            self.body_text.append(data)
        if self._block_tag:
            self._block_parts.append(data)

    def readable(self) -> tuple[str, str]:
        title = " ".join("".join(self.title_parts).split())
        selected = self.preferred_blocks or self.blocks
        if selected:
            unique: list[str] = []
            for block in selected:
                if not unique or block != unique[-1]:
                    unique.append(block)
            return title, "\n\n".join(unique)
        return title, " ".join("".join(self.body_text).split())


def extract_readable_html(html: str, fallback_title: str = "Web page") -> tuple[str, str]:
    parser = _ReadableHtmlParser()
    try:
        parser.feed(html)
        parser.close()
    except (UnicodeError, ValueError) as exc:
        raise InvalidDocument("The URL did not return readable HTML") from exc
    title, text = parser.readable()
    if not text.strip():
        raise InvalidDocument("The URL does not contain extractable text")
    return title or fallback_title, text.strip()


class WebDocumentFetcher:
    def __init__(
        self,
        *,
        timeout: float = 15.0,
        max_bytes: int = 5 * 1024 * 1024,
        max_redirects: int = 5,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.transport = transport

    @staticmethod
    def _public_address(address: str) -> bool:
        try:
            return ipaddress.ip_address(address).is_global
        except ValueError:
            return False

    async def _validate_url(self, url: str) -> str:
        parsed = urlsplit(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise InvalidDocument("Enter a valid public HTTP or HTTPS URL")
        if parsed.username or parsed.password:
            raise InvalidDocument("URLs containing credentials are not supported")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise InvalidDocument("Enter a valid public HTTP or HTTPS URL") from exc
        try:
            records = await asyncio.to_thread(
                socket.getaddrinfo,
                parsed.hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise InvalidDocument("The URL hostname could not be resolved") from exc
        addresses = {record[4][0] for record in records}
        if not addresses or any(not self._public_address(address) for address in addresses):
            raise InvalidDocument("Private or non-public network URLs are not supported")
        return urldefrag(parsed.geturl()).url

    async def fetch(self, url: str) -> WebPage:
        try:
            return await asyncio.wait_for(self._fetch(url), timeout=self.timeout)
        except TimeoutError as exc:
            raise InvalidDocument("The URL fetch timed out") from exc

    async def _fetch(self, url: str) -> WebPage:
        current_url = url
        headers = {
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "OmniRAG/1.0 knowledge-source importer",
        }
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=False,
            transport=self.transport,
        ) as client:
            for redirect_count in range(self.max_redirects + 1):
                current_url = await self._validate_url(current_url)
                try:
                    async with client.stream("GET", current_url, headers=headers) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location:
                                raise InvalidDocument("The URL returned an invalid redirect")
                            if redirect_count >= self.max_redirects:
                                raise InvalidDocument("The URL exceeded the redirect limit")
                            current_url = urljoin(current_url, location)
                            continue
                        if response.status_code >= 400:
                            raise InvalidDocument(
                                f"The URL returned HTTP {response.status_code}"
                            )
                        content_type = response.headers.get("content-type", "").lower()
                        if not any(
                            supported in content_type
                            for supported in ("text/html", "application/xhtml+xml")
                        ):
                            raise InvalidDocument("The URL must return an HTML page")
                        content = bytearray()
                        async for chunk in response.aiter_bytes():
                            content.extend(chunk)
                            if len(content) > self.max_bytes:
                                raise InvalidDocument("The URL content exceeds the configured limit")
                        charset = "utf-8"
                        match = re.search(r"charset=([^;\s]+)", content_type)
                        if match:
                            charset = match.group(1).strip('"\'')
                        try:
                            html = bytes(content).decode(charset, errors="replace")
                        except LookupError:
                            html = bytes(content).decode("utf-8", errors="replace")
                        final_url = urldefrag(str(response.url)).url
                        fallback_title = urlsplit(final_url).hostname or "Web page"
                        title, text = extract_readable_html(html, fallback_title)
                        return WebPage(url=final_url, title=title[:255], text=text)
                except httpx.HTTPError as exc:
                    raise InvalidDocument("The URL could not be fetched") from exc
        raise InvalidDocument("The URL could not be fetched")
