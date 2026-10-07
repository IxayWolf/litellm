import hashlib
import json
import re
from typing import Final
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    MICROSOFT_365_COPILOT_TOKEN_CACHE_SAFETY_MARGIN_SECONDS,
    MICROSOFT_ENTRA_LOGIN_AUTHORITY,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.proxy.litellm_pre_call_utils import SecretFields


class Microsoft365CopilotError(BaseLLMException):
    pass


class _OAuthTokenResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    access_token: str
    expires_in: int = Field(gt=0)


class _OAuthErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    error: str | None = None
    error_description: str | None = None


_TOKEN_RESPONSE_ADAPTER: Final = TypeAdapter(_OAuthTokenResponse)
_ERROR_RESPONSE_ADAPTER: Final = TypeAdapter(_OAuthErrorResponse)
_RAW_HEADERS_ADAPTER: Final = TypeAdapter(dict[str, str])
_SECRET_FIELDS_ADAPTER: Final = TypeAdapter(SecretFields)
_OBO_TOKEN_CACHE: Final = InMemoryCache(max_size_in_memory=1000, default_ttl=600)


def _cache_key(tenant_id: str, client_id: str, client_secret: str, assertion: str) -> str:
    key_material: Final = json.dumps(
        (tenant_id, client_id, client_secret, assertion),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


def redact_sensitive_values(message: str, sensitive_values: tuple[str, ...]) -> str:
    redacted_values: Final = tuple(sorted((value for value in sensitive_values if value), key=len, reverse=True))
    if not redacted_values:
        return message
    pattern: Final = re.compile("|".join(re.escape(value) for value in redacted_values))
    return pattern.sub("[REDACTED]", message)


def _parse_oauth_error(response: httpx.Response, sensitive_values: tuple[str, ...]) -> str:
    try:
        error_response: Final = _ERROR_RESPONSE_ADAPTER.validate_python(response.json())
    except (ValidationError, ValueError):
        return redact_sensitive_values(
            "Microsoft Entra on-behalf-of exchange failed: unknown_error: invalid response",
            sensitive_values,
        )
    error: Final = error_response.error or "unknown_error"
    description: Final = error_response.error_description or "no description provided"
    return redact_sensitive_values(
        f"Microsoft Entra on-behalf-of exchange failed: {error}: {description}",
        sensitive_values,
    )


def _parse_oauth_token(
    response: httpx.Response,
    tenant_id: str,
    client_id: str,
    client_secret: str,
    assertion: str,
) -> str:
    try:
        token_response: Final = _TOKEN_RESPONSE_ADAPTER.validate_python(response.json())
    except (ValidationError, ValueError):
        raise Microsoft365CopilotError(
            status_code=502,
            message="Microsoft Entra on-behalf-of exchange returned an invalid token response",
        ) from None
    effective_ttl: Final = token_response.expires_in - MICROSOFT_365_COPILOT_TOKEN_CACHE_SAFETY_MARGIN_SECONDS
    if effective_ttl > 0:
        _OBO_TOKEN_CACHE.set_cache(
            key=_cache_key(tenant_id, client_id, client_secret, assertion),
            value=token_response.access_token,
            ttl=effective_ttl,
        )
    return token_response.access_token


def extract_caller_assertion(secret_fields: SecretFields | None) -> str | None:
    if secret_fields is None:
        return None
    raw_headers: Final[object] = secret_fields.get("raw_headers")
    try:
        headers: Final = _RAW_HEADERS_ADAPTER.validate_python(raw_headers)
    except ValidationError:
        return None
    authorization_values: Final = tuple(value for name, value in headers.items() if name.casefold() == "authorization")
    if len(authorization_values) != 1:
        return None
    authorization_parts: Final = authorization_values[0].split(maxsplit=1)
    if len(authorization_parts) != 2 or authorization_parts[0].casefold() != "bearer":
        return None
    assertion: Final = authorization_parts[1].strip()
    assertion_parts: Final = tuple(assertion.split("."))
    if (
        len(assertion_parts) != 3
        or any(not part for part in assertion_parts)
        or any(character.isspace() for character in assertion)
    ):
        return None
    return assertion


def as_secret_fields(secret_fields: object) -> SecretFields | None:
    try:
        return _SECRET_FIELDS_ADAPTER.validate_python(secret_fields)
    except ValidationError:
        return None


def _token_request_data(client_id: str, client_secret: str, assertion: str) -> dict[str, str]:
    return {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "client_id": client_id,
        "client_secret": client_secret,
        "assertion": assertion,
        "scope": "https://graph.microsoft.com/.default",
        "requested_token_use": "on_behalf_of",
    }


def _cached_token(tenant_id: str, client_id: str, client_secret: str, assertion: str) -> str | None:
    cached_value: Final = _OBO_TOKEN_CACHE.get_cache(key=_cache_key(tenant_id, client_id, client_secret, assertion))
    return cached_value if isinstance(cached_value, str) else None


def _post_obo_request(
    client: HTTPHandler,
    token_url: str,
    request_data: dict[str, str],
    timeout: float | httpx.Timeout | None,
) -> httpx.Response:
    try:
        return client.post(token_url, data=request_data, timeout=timeout)
    except httpx.HTTPStatusError as error:
        return error.response
    except httpx.HTTPError:
        raise Microsoft365CopilotError(
            status_code=502,
            message="Microsoft Entra on-behalf-of exchange request failed",
        ) from None


async def _apost_obo_request(
    client: AsyncHTTPHandler,
    token_url: str,
    request_data: dict[str, str],
    timeout: float | httpx.Timeout | None,
) -> httpx.Response:
    try:
        return await client.post(token_url, data=request_data, timeout=timeout)
    except httpx.HTTPStatusError as error:
        return error.response
    except httpx.HTTPError:
        raise Microsoft365CopilotError(
            status_code=502,
            message="Microsoft Entra on-behalf-of exchange request failed",
        ) from None


def _obo_error_status_code(status_code: int) -> int:
    if status_code == 429:
        return status_code
    if 400 <= status_code < 500:
        return 401
    return status_code


def exchange_entra_token(
    client: HTTPHandler,
    tenant_id: str,
    client_id: str,
    client_secret: str,
    assertion: str,
    timeout: float | httpx.Timeout | None = None,
) -> str:
    cached_token: Final = _cached_token(tenant_id, client_id, client_secret, assertion)
    if cached_token is not None:
        return cached_token
    token_url: Final = f"{MICROSOFT_ENTRA_LOGIN_AUTHORITY}/{quote(tenant_id, safe='')}/oauth2/v2.0/token"
    response: Final = _post_obo_request(
        client=client,
        token_url=token_url,
        request_data=_token_request_data(client_id, client_secret, assertion),
        timeout=timeout,
    )
    if response.status_code != 200:
        error_message: Final = _parse_oauth_error(
            response=response,
            sensitive_values=(client_secret, assertion),
        )
        raise Microsoft365CopilotError(
            status_code=_obo_error_status_code(response.status_code),
            message=error_message,
        )
    return _parse_oauth_token(
        response=response,
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        assertion=assertion,
    )


async def aexchange_entra_token(
    client: AsyncHTTPHandler,
    tenant_id: str,
    client_id: str,
    client_secret: str,
    assertion: str,
    timeout: float | httpx.Timeout | None = None,
) -> str:
    cached_token: Final = _cached_token(tenant_id, client_id, client_secret, assertion)
    if cached_token is not None:
        return cached_token
    token_url: Final = f"{MICROSOFT_ENTRA_LOGIN_AUTHORITY}/{quote(tenant_id, safe='')}/oauth2/v2.0/token"
    response: Final = await _apost_obo_request(
        client=client,
        token_url=token_url,
        request_data=_token_request_data(client_id, client_secret, assertion),
        timeout=timeout,
    )
    if response.status_code != 200:
        error_message: Final = _parse_oauth_error(
            response=response,
            sensitive_values=(client_secret, assertion),
        )
        raise Microsoft365CopilotError(
            status_code=_obo_error_status_code(response.status_code),
            message=error_message,
        )
    return _parse_oauth_token(
        response=response,
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        assertion=assertion,
    )
