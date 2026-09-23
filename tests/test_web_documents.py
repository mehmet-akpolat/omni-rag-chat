import asyncio
import socket

import httpx
import pytest

from packages.omni_rag_core.documents import InvalidDocument
from packages.omni_rag_core.web_documents import WebDocumentFetcher, extract_readable_html


class PublicTestFetcher(WebDocumentFetcher):
    async def _validate_url(self, url: str) -> str:
        return url


def test_extract_readable_html_prefers_main_content_and_preserves_headings():
    title, text = extract_readable_html(
        """
        <html><head><title> Product   Guide </title><script>bad()</script></head>
        <body><nav>Menu text</nav><p>Outside article.</p><main>
        <h1>Getting Started</h1><p>Install the product safely.</p>
        <h2>Next step</h2><p>Open <strong>Settings</strong> to continue.</p>
        </main><footer>Copyright</footer></body></html>
        """
    )
    assert title == "Product Guide"
    assert text == (
        "# Getting Started\n\nInstall the product safely.\n\n"
        "## Next step\n\nOpen Settings to continue."
    )
    assert "Menu" not in text and "Outside" not in text and "Copyright" not in text


def test_extract_readable_html_falls_back_to_body_text_and_rejects_empty_content():
    assert extract_readable_html("<body>Loose <span>body</span> text.</body>", "Fallback") == (
        "Fallback",
        "Loose body text.",
    )
    with pytest.raises(InvalidDocument, match="extractable text"):
        extract_readable_html("<html><title>Empty</title><nav>Only navigation</nav></html>")


def test_url_validation_rejects_unsafe_addresses_and_credentials(monkeypatch):
    fetcher = WebDocumentFetcher()
    assert fetcher._public_address("8.8.8.8") is True
    assert fetcher._public_address("127.0.0.1") is False
    assert fetcher._public_address("invalid") is False
    with pytest.raises(InvalidDocument, match="valid public"):
        asyncio.run(fetcher._validate_url("file:///etc/passwd"))
    with pytest.raises(InvalidDocument, match="credentials"):
        asyncio.run(fetcher._validate_url("https://user:secret@example.com"))
    with pytest.raises(InvalidDocument, match="valid public"):
        asyncio.run(fetcher._validate_url("https://example.com:invalid"))

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
    )
    with pytest.raises(InvalidDocument, match="Private"):
        asyncio.run(fetcher._validate_url("http://example.com"))

    def unresolved(*args, **kwargs):
        raise socket.gaierror

    monkeypatch.setattr(socket, "getaddrinfo", unresolved)
    with pytest.raises(InvalidDocument, match="resolved"):
        asyncio.run(fetcher._validate_url("https://missing.example"))


def test_fetch_follows_safe_redirect_and_extracts_html():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/guide"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<title>Guide</title><main><h1>Help</h1><p>Useful answer.</p></main>",
        )

    fetcher = PublicTestFetcher(transport=httpx.MockTransport(handler))
    page = asyncio.run(fetcher.fetch("https://example.com/start"))
    assert page.url == "https://example.com/guide"
    assert page.title == "Guide"
    assert page.text == "# Help\n\nUseful answer."


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(404, headers={"content-type": "text/html"}), "HTTP 404"),
        (httpx.Response(200, headers={"content-type": "application/json"}), "HTML page"),
        (
            httpx.Response(
                302,
                headers={"location": "https://example.com/again"},
            ),
            "redirect limit",
        ),
    ],
)
def test_fetch_rejects_invalid_responses(response: httpx.Response, message: str):
    fetcher = PublicTestFetcher(
        max_redirects=0,
        transport=httpx.MockTransport(lambda request: response),
    )
    with pytest.raises(InvalidDocument, match=message):
        asyncio.run(fetcher.fetch("https://example.com/start"))


def test_fetch_enforces_size_and_wraps_network_errors():
    oversized = PublicTestFetcher(
        max_bytes=10,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"<p>This response is too large</p>",
            )
        ),
    )
    with pytest.raises(InvalidDocument, match="configured limit"):
        asyncio.run(oversized.fetch("https://example.com"))

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    unavailable = PublicTestFetcher(transport=httpx.MockTransport(fail))
    with pytest.raises(InvalidDocument, match="could not be fetched"):
        asyncio.run(unavailable.fetch("https://example.com"))


def test_fetch_timeout_covers_the_complete_operation():
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.02)
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<p>Late</p>")

    fetcher = PublicTestFetcher(timeout=0.001, transport=httpx.MockTransport(slow))
    with pytest.raises(InvalidDocument, match="timed out"):
        asyncio.run(fetcher.fetch("https://example.com"))
