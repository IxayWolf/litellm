from collections.abc import Callable
from typing import Final
from urllib.parse import parse_qs

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.microsoft_365_copilot.common_utils import (
    Microsoft365CopilotError,
    aexchange_entra_token,
    exchange_entra_token,
    extract_caller_assertion,
)
from litellm.types.proxy.litellm_pre_call_utils import SecretFields

_OBO_ERROR_STATUS_CASES: Final = (
    (400, "invalid_grant", 401),
    (429, "too_many_requests", 429),
)


class _AsyncCaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self, responder: Callable[[httpx.Request], httpx.Response]) -> None:
        self._responder = responder

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._responder(request)


def _sync_client(
    oauth_status: int,
    oauth_payload: object,
) -> tuple[HTTPHandler, list[httpx.Request]]:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=oauth_status,
            json=oauth_payload,
            request=request,
        )

    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    return client, requests


def test_sync_obo_exchange_sends_exact_form_and_returns_graph_token() -> None:
    client, requests = _sync_client(
        oauth_status=200,
        oauth_payload={"access_token": "graph-token-sync", "expires_in": 3600},
    )

    token: Final = exchange_entra_token(
        client=client,
        tenant_id="tenant-sync-9306",
        client_id="client-sync-9306",
        client_secret="secret-sync-9306",
        assertion="header-sync.payload-sync.signature-sync",
    )

    form: Final = {name: values[0] for name, values in parse_qs(requests[0].content.decode("utf-8")).items()}
    assert str(requests[0].url) == ("https://login.microsoftonline.com/tenant-sync-9306/oauth2/v2.0/token")
    assert form == {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "client_id": "client-sync-9306",
        "client_secret": "secret-sync-9306",
        "assertion": "header-sync.payload-sync.signature-sync",
        "scope": "https://graph.microsoft.com/.default",
        "requested_token_use": "on_behalf_of",
    }
    assert token == "graph-token-sync"


@pytest.mark.asyncio
async def test_async_obo_exchange_sends_exact_form_and_returns_graph_token() -> None:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={"access_token": "graph-token-async", "expires_in": 3600},
            request=request,
        )

    client: Final = AsyncHTTPHandler(transport=_AsyncCaptureTransport(respond))
    async with client.client:
        token: Final = await aexchange_entra_token(
            client=client,
            tenant_id="tenant-async-9306",
            client_id="client-async-9306",
            client_secret="secret-async-9306",
            assertion="header-async.payload-async.signature-async",
        )

    form: Final = {name: values[0] for name, values in parse_qs(requests[0].content.decode("utf-8")).items()}
    assert str(requests[0].url) == ("https://login.microsoftonline.com/tenant-async-9306/oauth2/v2.0/token")
    assert form == {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "client_id": "client-async-9306",
        "client_secret": "secret-async-9306",
        "assertion": "header-async.payload-async.signature-async",
        "scope": "https://graph.microsoft.com/.default",
        "requested_token_use": "on_behalf_of",
    }
    assert token == "graph-token-async"


def test_obo_exchange_maps_entra_4xx_and_redacts_credentials() -> None:
    client_secret: Final = "secret-error-9306"
    assertion: Final = "header-error.payload-error.signature-error"
    client, _ = _sync_client(
        oauth_status=400,
        oauth_payload={
            "error": "invalid_grant",
            "error_description": f"assertion {assertion} secret {client_secret}",
        },
    )

    with pytest.raises(Microsoft365CopilotError) as error:
        exchange_entra_token(
            client=client,
            tenant_id="tenant-error-9306",
            client_id="client-error-9306",
            client_secret=client_secret,
            assertion=assertion,
        )

    assert error.value.status_code == 401
    assert "invalid_grant" in str(error.value)
    assert client_secret not in str(error.value)
    assert assertion not in str(error.value)


@pytest.mark.parametrize(
    ("oauth_status", "oauth_error", "expected_status_code"),
    _OBO_ERROR_STATUS_CASES,
)
def test_sync_obo_exchange_maps_entra_429_and_invalid_grant(
    oauth_status: int,
    oauth_error: str,
    expected_status_code: int,
) -> None:
    client, _ = _sync_client(
        oauth_status=oauth_status,
        oauth_payload={"error": oauth_error, "error_description": "request failed"},
    )

    with pytest.raises(Microsoft365CopilotError) as error:
        exchange_entra_token(
            client=client,
            tenant_id="tenant-status-9306",
            client_id="client-status-9306",
            client_secret="secret-status-9306",
            assertion="header-status.payload-status.signature-status",
        )

    assert error.value.status_code == expected_status_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("oauth_status", "oauth_error", "expected_status_code"),
    _OBO_ERROR_STATUS_CASES,
)
async def test_async_obo_exchange_maps_entra_429_and_invalid_grant(
    oauth_status: int,
    oauth_error: str,
    expected_status_code: int,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=oauth_status,
            json={"error": oauth_error, "error_description": "request failed"},
            request=request,
        )

    client: Final = AsyncHTTPHandler(transport=_AsyncCaptureTransport(respond))
    async with client.client:
        with pytest.raises(Microsoft365CopilotError) as error:
            await aexchange_entra_token(
                client=client,
                tenant_id="tenant-status-9306",
                client_id="client-status-9306",
                client_secret="secret-status-9306",
                assertion="header-status.payload-status.signature-status",
            )

    assert error.value.status_code == expected_status_code


def test_obo_exchange_preserves_non_4xx_status() -> None:
    client, _ = _sync_client(
        oauth_status=500,
        oauth_payload={"error": "server_error", "error_description": "temporary failure"},
    )

    with pytest.raises(Microsoft365CopilotError) as error:
        exchange_entra_token(
            client=client,
            tenant_id="tenant-server-error-9306",
            client_id="client-server-error-9306",
            client_secret="secret-server-error-9306",
            assertion="header-server-error.payload-server-error.signature-server-error",
        )

    assert error.value.status_code == 500
    assert str(error.value) == ("Microsoft Entra on-behalf-of exchange failed: server_error: temporary failure")


def test_obo_exchange_does_not_cache_tokens_inside_safety_margin() -> None:
    client, requests = _sync_client(
        oauth_status=200,
        oauth_payload={"access_token": "short-lived-token-9306", "expires_in": 30},
    )
    exchange_params: Final = {
        "client": client,
        "tenant_id": "tenant-short-9306",
        "client_id": "client-short-9306",
        "client_secret": "secret-short-9306",
        "assertion": "header-short.payload-short.signature-short",
    }

    first_token: Final = exchange_entra_token(**exchange_params)
    second_token: Final = exchange_entra_token(**exchange_params)

    assert first_token == "short-lived-token-9306"
    assert second_token == first_token
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("authorization", "expected"),
    [
        ("bEaReR header.payload.signature", "header.payload.signature"),
        ("Bearer header..signature", None),
        ("Bearer not-a-jwt", None),
        ("Basic header.payload.signature", None),
        ("Bearer header.payload.signature extra", None),
    ],
)
def test_extract_caller_assertion_requires_a_three_part_bearer_jwt(
    authorization: str,
    expected: str | None,
) -> None:
    secret_fields: Final[SecretFields] = SecretFields(raw_headers={"aUtHoRiZaTiOn": authorization})

    assertion: Final = extract_caller_assertion(secret_fields)

    assert assertion == expected
