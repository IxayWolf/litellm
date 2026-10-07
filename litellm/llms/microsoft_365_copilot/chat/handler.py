from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final
from urllib.parse import quote

import httpx
from aiohttp import ClientSession

from litellm import LlmProviders
from litellm.constants import MICROSOFT_GRAPH_BETA_BASE
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
    get_httpx_client,
)
from litellm.llms.custom_httpx.llm_http_handler import MockResponseIterator
from litellm.llms.microsoft_365_copilot.chat.transformation import (
    GraphChatRequest,
    build_chat_request,
    extract_graph_error_message,
    map_graph_response,
    parse_graph_conversation_id,
)
from litellm.llms.microsoft_365_copilot.common_utils import (
    Microsoft365CopilotError,
    aexchange_entra_token,
    exchange_entra_token,
    extract_caller_assertion,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.litellm_pre_call_utils import SecretFields
from litellm.types.utils import ModelResponse


@dataclass(frozen=True, slots=True, repr=False)
class _OBOCredentials:
    tenant_id: str
    client_id: str
    client_secret: str
    assertion: str


def _get_obo_credentials(
    litellm_params: Mapping[str, object],
    secret_fields: SecretFields | None,
) -> _OBOCredentials | None:
    tenant_id: Final[object] = litellm_params.get("tenant_id")
    client_id: Final[object] = litellm_params.get("client_id")
    client_secret: Final[object] = litellm_params.get("client_secret")
    credential_values: Final = (tenant_id, client_id, client_secret)
    configured_values: Final = tuple(value is not None and value != "" for value in credential_values)
    if not any(configured_values):
        return None
    if not all(configured_values):
        raise Microsoft365CopilotError(
            status_code=400,
            message="microsoft_365_copilot requires tenant_id, client_id, and client_secret together",
        )
    if not isinstance(tenant_id, str) or not isinstance(client_id, str) or not isinstance(client_secret, str):
        raise Microsoft365CopilotError(
            status_code=400,
            message="microsoft_365_copilot requires tenant_id, client_id, and client_secret as non-empty strings",
        )
    if not tenant_id or not client_id or not client_secret:
        raise Microsoft365CopilotError(
            status_code=400,
            message="microsoft_365_copilot requires tenant_id, client_id, and client_secret together",
        )
    assertion: Final = extract_caller_assertion(secret_fields)
    if assertion is None:
        raise Microsoft365CopilotError(
            status_code=401,
            message=(
                "microsoft_365_copilot with Entra on-behalf-of requires the caller's Microsoft Entra user "
                "access token in the Authorization header"
            ),
        )
    return _OBOCredentials(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        assertion=assertion,
    )


def _resolve_access_token(
    client: HTTPHandler,
    api_key: str | None,
    litellm_params: Mapping[str, object],
    secret_fields: SecretFields | None,
    timeout: float | httpx.Timeout | None,
) -> str:
    obo_credentials: Final = _get_obo_credentials(
        litellm_params=litellm_params,
        secret_fields=secret_fields,
    )
    if obo_credentials is not None:
        return exchange_entra_token(
            client=client,
            tenant_id=obo_credentials.tenant_id,
            client_id=obo_credentials.client_id,
            client_secret=obo_credentials.client_secret,
            assertion=obo_credentials.assertion,
            timeout=timeout,
        )
    if isinstance(api_key, str) and api_key:
        return api_key
    raise Microsoft365CopilotError(
        status_code=401,
        message=(
            "microsoft_365_copilot requires tenant_id, client_id, and client_secret with a caller "
            "Authorization header, or a delegated Graph api_key"
        ),
    )


async def _aresolve_access_token(
    client: AsyncHTTPHandler,
    api_key: str | None,
    litellm_params: Mapping[str, object],
    secret_fields: SecretFields | None,
    timeout: float | httpx.Timeout | None,
) -> str:
    obo_credentials: Final = _get_obo_credentials(
        litellm_params=litellm_params,
        secret_fields=secret_fields,
    )
    if obo_credentials is not None:
        return await aexchange_entra_token(
            client=client,
            tenant_id=obo_credentials.tenant_id,
            client_id=obo_credentials.client_id,
            client_secret=obo_credentials.client_secret,
            assertion=obo_credentials.assertion,
            timeout=timeout,
        )
    if isinstance(api_key, str) and api_key:
        return api_key
    raise Microsoft365CopilotError(
        status_code=401,
        message=(
            "microsoft_365_copilot requires tenant_id, client_id, and client_secret with a caller "
            "Authorization header, or a delegated Graph api_key"
        ),
    )


def _json_body(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return {}


def _graph_error(
    response: httpx.Response,
    sensitive_values: tuple[str, ...],
) -> Microsoft365CopilotError:
    return Microsoft365CopilotError(
        status_code=response.status_code,
        message=extract_graph_error_message(
            response_body=_json_body(response),
            sensitive_values=sensitive_values,
        ),
    )


def _pre_call(
    logging_obj: Logging | None,
    messages: Sequence[AllMessageValues],
    model: str,
    request_data: GraphChatRequest,
) -> None:
    if logging_obj is None:
        return
    logging_obj.pre_call(
        input=messages,
        api_key="",
        model=model,
        additional_args={
            "api_base": MICROSOFT_GRAPH_BETA_BASE,
            "complete_input_dict": request_data,
        },
    )


def _post_call(
    logging_obj: Logging | None,
    messages: Sequence[AllMessageValues],
    request_data: GraphChatRequest,
    status_code: int,
) -> None:
    if logging_obj is None:
        return
    logging_obj.post_call(
        original_response={"status_code": status_code},
        input=messages,
        api_key="",
        additional_args={
            "api_base": MICROSOFT_GRAPH_BETA_BASE,
            "complete_input_dict": request_data,
        },
    )


def _post_graph_request(
    client: HTTPHandler,
    url: str,
    access_token: str,
    request_data: GraphChatRequest | dict[str, object],
    timeout: float | httpx.Timeout | None,
) -> httpx.Response:
    try:
        return client.post(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            json=request_data,
            timeout=timeout,
        )
    except httpx.HTTPStatusError as error:
        return error.response
    except httpx.HTTPError:
        raise Microsoft365CopilotError(status_code=502, message="Microsoft Graph request failed") from None


async def _apost_graph_request(
    client: AsyncHTTPHandler,
    url: str,
    access_token: str,
    request_data: GraphChatRequest | dict[str, object],
    timeout: float | httpx.Timeout | None,
) -> httpx.Response:
    try:
        return await client.post(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            json=request_data,
            timeout=timeout,
        )
    except httpx.HTTPStatusError as error:
        return error.response
    except httpx.HTTPError:
        raise Microsoft365CopilotError(status_code=502, message="Microsoft Graph request failed") from None


def _stream_response(
    response: ModelResponse,
    model: str,
    logging_obj: Logging | None,
) -> CustomStreamWrapper:
    return CustomStreamWrapper(
        completion_stream=MockResponseIterator(model_response=response),
        model=model,
        custom_llm_provider=LlmProviders.MICROSOFT_365_COPILOT.value,
        logging_obj=logging_obj,
    )


def _raise_graph_error_if_needed(
    response: httpx.Response,
    sensitive_values: tuple[str, ...],
    logging_obj: Logging | None,
    messages: Sequence[AllMessageValues],
    request_data: GraphChatRequest,
) -> None:
    if response.is_success:
        return
    _post_call(
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
        status_code=response.status_code,
    )
    raise _graph_error(response=response, sensitive_values=sensitive_values)


def _timeout_value(timeout: float | str | httpx.Timeout | None) -> float | httpx.Timeout | None:
    if not isinstance(timeout, str):
        return timeout
    try:
        return float(timeout)
    except ValueError:
        raise Microsoft365CopilotError(status_code=400, message="timeout must be a number") from None


def completion(
    model: str,
    messages: Sequence[AllMessageValues],
    api_key: str | None,
    litellm_params: Mapping[str, object],
    optional_params: Mapping[str, object],
    secret_fields: SecretFields | None = None,
    stream: bool = False,
    timeout: float | str | httpx.Timeout | None = None,
    logging_obj: Logging | None = None,
    client: HTTPHandler | None = None,
) -> ModelResponse | CustomStreamWrapper:
    request_data: Final = build_chat_request(
        messages=messages,
        optional_params=optional_params,
    )
    http_client: Final = client if client is not None else get_httpx_client(params={})
    timeout_value: Final = _timeout_value(timeout)
    _pre_call(logging_obj=logging_obj, messages=messages, model=model, request_data=request_data)
    access_token: Final = _resolve_access_token(
        client=http_client,
        api_key=api_key,
        litellm_params=litellm_params,
        secret_fields=secret_fields,
        timeout=timeout_value,
    )
    assertion: Final = extract_caller_assertion(secret_fields)
    configured_client_secret: Final[object] = litellm_params.get("client_secret")
    sensitive_client_secret: Final = configured_client_secret if isinstance(configured_client_secret, str) else ""
    sensitive_values: Final = (access_token, api_key or "", assertion or "", sensitive_client_secret)
    conversation_url: Final = f"{MICROSOFT_GRAPH_BETA_BASE}/copilot/conversations"
    conversation_response: Final = _post_graph_request(
        client=http_client,
        url=conversation_url,
        access_token=access_token,
        request_data={},
        timeout=timeout_value,
    )
    _raise_graph_error_if_needed(
        response=conversation_response,
        sensitive_values=sensitive_values,
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
    )
    conversation_id: Final = parse_graph_conversation_id(_json_body(conversation_response))
    chat_url: Final = f"{conversation_url}/{quote(conversation_id, safe='')}/chat"
    chat_response: Final = _post_graph_request(
        client=http_client,
        url=chat_url,
        access_token=access_token,
        request_data=request_data,
        timeout=timeout_value,
    )
    _raise_graph_error_if_needed(
        response=chat_response,
        sensitive_values=sensitive_values,
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
    )
    _post_call(
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
        status_code=chat_response.status_code,
    )
    response: Final = map_graph_response(graph_response=_json_body(chat_response), model=model)
    return _stream_response(response, model, logging_obj) if stream else response


async def acompletion(
    model: str,
    messages: Sequence[AllMessageValues],
    api_key: str | None,
    litellm_params: Mapping[str, object],
    optional_params: Mapping[str, object],
    secret_fields: SecretFields | None = None,
    stream: bool = False,
    timeout: float | str | httpx.Timeout | None = None,
    logging_obj: Logging | None = None,
    client: AsyncHTTPHandler | None = None,
    shared_session: ClientSession | None = None,
) -> ModelResponse | CustomStreamWrapper:
    request_data: Final = build_chat_request(
        messages=messages,
        optional_params=optional_params,
    )
    http_client: Final = (
        client
        if client is not None
        else get_async_httpx_client(
            llm_provider=LlmProviders.MICROSOFT_365_COPILOT,
            params={},
            shared_session=shared_session,
        )
    )
    timeout_value: Final = _timeout_value(timeout)
    _pre_call(logging_obj=logging_obj, messages=messages, model=model, request_data=request_data)
    access_token: Final = await _aresolve_access_token(
        client=http_client,
        api_key=api_key,
        litellm_params=litellm_params,
        secret_fields=secret_fields,
        timeout=timeout_value,
    )
    assertion: Final = extract_caller_assertion(secret_fields)
    configured_client_secret: Final[object] = litellm_params.get("client_secret")
    sensitive_client_secret: Final = configured_client_secret if isinstance(configured_client_secret, str) else ""
    sensitive_values: Final = (access_token, api_key or "", assertion or "", sensitive_client_secret)
    conversation_url: Final = f"{MICROSOFT_GRAPH_BETA_BASE}/copilot/conversations"
    conversation_response: Final = await _apost_graph_request(
        client=http_client,
        url=conversation_url,
        access_token=access_token,
        request_data={},
        timeout=timeout_value,
    )
    _raise_graph_error_if_needed(
        response=conversation_response,
        sensitive_values=sensitive_values,
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
    )
    conversation_id: Final = parse_graph_conversation_id(_json_body(conversation_response))
    chat_url: Final = f"{conversation_url}/{quote(conversation_id, safe='')}/chat"
    chat_response: Final = await _apost_graph_request(
        client=http_client,
        url=chat_url,
        access_token=access_token,
        request_data=request_data,
        timeout=timeout_value,
    )
    _raise_graph_error_if_needed(
        response=chat_response,
        sensitive_values=sensitive_values,
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
    )
    _post_call(
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
        status_code=chat_response.status_code,
    )
    response: Final = map_graph_response(graph_response=_json_body(chat_response), model=model)
    return _stream_response(response, model, logging_obj) if stream else response
