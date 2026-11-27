import base64
import json

import httpx
import pytest

from filesonar.config import ModelConfig
from filesonar.models import (
    CAPTION_PROMPT,
    ModelError,
    OllamaCaptioner,
    OllamaClient,
    OllamaEmbedder,
)


def client_with(handler, **cfg):
    config = ModelConfig(**cfg)
    http = httpx.Client(base_url=config.host, transport=httpx.MockTransport(handler))
    return OllamaClient(config, client=http)


def test_embeddings_are_batched_and_prefixed():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * len(body["input"])})

    emb = OllamaEmbedder(client_with(handler), batch_size=2)
    out = emb.embed_documents(["a", "b", "c"])
    assert len(out) == 3 and len(seen) == 2
    assert seen[0]["input"] == ["search_document: a", "search_document: b"]
    assert seen[0]["model"] == "nomic-embed-text"
    emb.embed_query("q")
    assert seen[-1]["input"] == ["search_query: q"]


def test_unknown_embedding_models_get_no_prefix():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    OllamaEmbedder(client_with(handler, embedding_model="mxbai-embed-large")).embed_query("q")
    assert seen[0]["input"] == ["q"]


@pytest.mark.parametrize(
    "response, message",
    [
        (httpx.Response(500, text="boom"), "500"),
        (httpx.Response(404, text="model not found"), "not found"),
        (httpx.Response(200, text="not json"), "invalid JSON"),
        (httpx.Response(200, json={"embeddings": []}), "did not match"),
    ],
)
def test_bad_responses_raise_model_error(response, message):
    emb = OllamaEmbedder(client_with(lambda r: response))
    with pytest.raises(ModelError, match=message):
        emb.embed_query("q")


def test_connection_error_raises_model_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ModelError, match="cannot reach"):
        OllamaEmbedder(client_with(handler)).embed_query("q")


def test_default_client_ignores_proxy_environment(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://attacker.example:8080")
    client = OllamaClient(ModelConfig())
    assert client.http._trust_env is False


def test_caption_request_and_cleanup():
    seen = {}

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5vl:7b"}]})
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "  A cat\n on a  sofa. "}})

    cap = OllamaCaptioner(client_with(handler))
    assert cap.available()
    assert cap.caption(b"\x89PNGfake") == "A cat on a sofa."
    msg = seen["messages"][0]
    assert msg["content"] == CAPTION_PROMPT
    assert base64.b64decode(msg["images"][0]) == b"\x89PNGfake"
    assert seen["stream"] is False


def test_captioner_availability_handles_latest_tag_and_errors():
    tags = httpx.Response(200, json={"models": [{"name": "llava:latest"}]})
    assert OllamaCaptioner(client_with(lambda r: tags, vision_model="llava")).available()
    assert not OllamaCaptioner(client_with(lambda r: tags)).available()
    broken = OllamaCaptioner(client_with(lambda r: httpx.Response(500)))
    assert not broken.available()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("JIACHE JIACHE JIACHE JIACHE JIACHE", "JIACHE"),
        ("A sticker. JIACHE JIACHE JIACHE JIACHE", "A sticker. JIACHE"),
        ("a red car, a red car, a red car, a red car", "a red car"),
        ("the the cat", "the the cat"),  # two in a row is normal language
        ("Step 1, step 2, step 3", "Step 1, step 2, step 3"),
    ],
)
def test_collapse_repeats(raw, expected):
    from filesonar.models import collapse_repeats

    assert collapse_repeats(raw) == expected


def test_runaway_caption_is_cleaned_and_generation_is_capped():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "Logo text " + "JIACHE " * 400}})

    text = OllamaCaptioner(client_with(handler)).caption(b"\x89PNG")
    assert text == "Logo text JIACHE"
    assert seen["options"]["num_predict"] <= 300
    assert seen["options"]["repeat_penalty"] > 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The prompt's own examples echoed back as negatives.
        (
            "The image is a logo design featuring a gift box. The image is a digital "
            "illustration, not a photo, screenshot, scanned document, receipt, or diagram.",
            "The image is a logo design featuring a gift box. The image is a digital illustration.",
        ),
        # A sentence that only reports absence is dropped.
        (
            "A white arrow on a black background. There is no visible text, brands, or "
            "numbers within this image. The design is simple.",
            "A white arrow on a black background. The design is simple.",
        ),
        # The informative start of a sentence is kept.
        (
            "The image is a completely black screen with no discernible content such as "
            "photos or receipts. It looks like a loading error.",
            "The image is a completely black screen. It looks like a loading error.",
        ),
        # Real content that happens to contain "no" stays.
        (
            "A receipt from Home Depot showing 3 items. No returns are accepted after 90 days.",
            "A receipt from Home Depot showing 3 items. No returns are accepted after 90 days.",
        ),
        ("A screenshot of a chat. There is no text visible.", "A screenshot of a chat."),
        # Never return nothing.
        ("There is no visible content.", "There is no visible content."),
    ],
)
def test_drop_absence_sentences(raw, expected):
    from filesonar.models import drop_absence_sentences

    assert drop_absence_sentences(raw) == expected


def test_caption_prompt_lists_no_categories():
    """Listing example categories made the model echo them ("not a receipt, diagram")."""
    from filesonar.models import CAPTION_PROMPT

    assert "receipt" not in CAPTION_PROMPT and "diagram" not in CAPTION_PROMPT
    assert "absent" in CAPTION_PROMPT and "instructions" in CAPTION_PROMPT


def test_queries_use_a_short_timeout_documents_the_configured_one():
    """Found with a hung Ollama: a search waited the full 120 s caption timeout."""
    from filesonar.models import QUERY_TIMEOUT_SECONDS

    timeouts = []

    def handler(request):
        timeouts.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    config = ModelConfig(timeout_seconds=120)
    real_timeouts = OllamaClient(config).http.timeout  # what filesonar really configures
    http = httpx.Client(
        base_url=config.host, transport=httpx.MockTransport(handler), timeout=real_timeouts
    )
    emb = OllamaEmbedder(OllamaClient(config, client=http))
    emb.embed_query("q")
    emb.embed_documents(["d"])
    assert timeouts == [QUERY_TIMEOUT_SECONDS, 120]  # documents keep the long limit
    short = OllamaEmbedder(client_with(handler, timeout_seconds=5))
    short.embed_query("q")
    assert timeouts[-1] == 5  # never longer than the configured limit


def test_tests_never_reach_the_real_ollama():
    """The suite must not depend on (or disturb) an Ollama running on this machine."""
    assert OllamaClient(ModelConfig()).version() is None
    assert OllamaClient(ModelConfig()).loaded_models() == {}
