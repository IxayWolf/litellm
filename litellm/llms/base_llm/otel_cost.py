from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from math import isfinite, nan
from typing import Annotated, Final

from pydantic import AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

import litellm
from litellm.exceptions import ModelNotMappedError
from litellm.integrations.otel.model.semconv import litellm_provider_names, resolve_provider
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import (
    CacheCreationTokenDetails,
    CompletionTokensDetailsWrapper,
    ModelInfo,
    ModelInfoBase,
    PromptTokensDetailsWrapper,
    Usage,
)
from litellm.utils import get_model_info_helper

_INPUT_TOKEN_KEYS: Final = ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens")
_OUTPUT_TOKEN_KEYS: Final = ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens")
_CACHE_WRITE_KEYS: Final = ("gen_ai.usage.cache_write.input_tokens", "gen_ai.usage.cache_creation.input_tokens")
_PROVIDER_KEYS: Final = ("gen_ai.provider.name", "gen_ai.system")
_RESPONSE_TIER_KEYS: Final = (
    "openai.response.service_tier",
    "anthropic.response.service_tier",
    "gen_ai.openai.response.service_tier",
)
_REQUEST_TIER_KEYS: Final = ("openai.request.service_tier", "gen_ai.openai.request.service_tier")


def _token_count(value: object) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError("Expected an unsigned token count")
    return int(value)


TokenCount = Annotated[int, BeforeValidator(_token_count), Field(ge=0, le=18446744073709551615)]


def _timestamp_ns(value: object) -> int:
    return -_token_count(value[1:]) if isinstance(value, str) and value.startswith("-") else _token_count(value)


TimestampNs = Annotated[int, BeforeValidator(_timestamp_ns), Field(ge=-9223372036854775808, le=9223372036854775807)]


class TraceUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    model: str = Field(min_length=1, validation_alias=AliasChoices("gen_ai.response.model", "gen_ai.request.model"))
    provider: str | None = Field(default=None, validation_alias=_PROVIDER_KEYS[0])
    legacy_provider: str | None = Field(default=None, validation_alias=_PROVIDER_KEYS[1])
    input_tokens: TokenCount = Field(validation_alias=AliasChoices(*_INPUT_TOKEN_KEYS))
    output_tokens: TokenCount = Field(validation_alias=AliasChoices(*_OUTPUT_TOKEN_KEYS))
    reasoning_tokens: TokenCount | None = Field(default=None, validation_alias="gen_ai.usage.reasoning.output_tokens")
    start_ns: TimestampNs | None = Field(default=None, validation_alias="litellm.trace.start_ns")
    cache_read_tokens: TokenCount = Field(default=0, validation_alias="gen_ai.usage.cache_read.input_tokens")
    cache_write_tokens: TokenCount = Field(default=0, validation_alias=AliasChoices(*_CACHE_WRITE_KEYS))
    cache_write_5m: TokenCount | None = Field(
        default=None, validation_alias="anthropic.usage.cache_creation.ephemeral_5m_input_tokens"
    )
    cache_write_1h: TokenCount | None = Field(
        default=None, validation_alias="anthropic.usage.cache_creation.ephemeral_1h_input_tokens"
    )
    response_service_tier: str | None = Field(default=None, validation_alias=AliasChoices(*_RESPONSE_TIER_KEYS))
    request_service_tier: str | None = Field(default=None, validation_alias=AliasChoices(*_REQUEST_TIER_KEYS))

    @model_validator(mode="before")
    @classmethod
    def consistent_aliases(cls, attributes: Mapping[str, str]) -> Mapping[str, str]:
        for keys in (_INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS, _CACHE_WRITE_KEYS):
            if len({_token_count(attributes[key]) for key in keys if key in attributes}) > 1:
                raise ValueError("Conflicting token usage aliases")
        for keys in (_PROVIDER_KEYS, _RESPONSE_TIER_KEYS, _REQUEST_TIER_KEYS):
            values: Final = {
                resolve_provider(attributes[key]) if keys == _PROVIDER_KEYS else attributes[key]
                for key in keys
                if key in attributes
            }
            if len(values) > 1:
                raise ValueError("Conflicting provider or service-tier aliases")
        return attributes

    @property
    def service_tier(self) -> str | None:
        return self.response_service_tier if self.response_service_tier is not None else self.request_service_tier

    @property
    def start_time(self) -> datetime | None:
        if self.start_ns is None:
            return None
        seconds, nanoseconds = divmod(self.start_ns, 1_000_000_000)
        return datetime.fromtimestamp(seconds, timezone.utc).replace(microsecond=nanoseconds // 1000)

    def provider_names(self) -> tuple[str | None, ...]:
        observed: Final = tuple(value for value in (self.provider, self.legacy_provider) if value is not None)
        if not observed:
            return (None,)
        candidates: Final = litellm_provider_names(observed[0], self.model)
        return tuple(
            candidate
            for candidate in candidates
            if all(candidate in litellm_provider_names(value, self.model) for value in observed[1:])
        )

    @model_validator(mode="after")
    def consistent_cache_usage(self) -> "TraceUsage":
        if self.reasoning_tokens is not None and self.reasoning_tokens > self.output_tokens:
            raise ValueError("Reasoning usage exceeds total output")
        if self.cache_read_tokens + self.cache_write_tokens > self.input_tokens:
            raise ValueError("Cached input exceeds total input")
        if self.cache_write_5m is not None or self.cache_write_1h is not None:
            if self.cache_write_5m is None or self.cache_write_1h is None:
                raise ValueError("Cache creation duration breakdown is incomplete")
            if self.cache_write_5m + self.cache_write_1h != self.cache_write_tokens:
                raise ValueError("Cache creation duration breakdown disagrees with total")
        return self

    def usage(self) -> Usage:
        details: Final = (
            CacheCreationTokenDetails(
                ephemeral_5m_input_tokens=self.cache_write_5m,
                ephemeral_1h_input_tokens=self.cache_write_1h,
            )
            if self.cache_write_5m is not None
            else None
        )
        return Usage(
            prompt_tokens=self.input_tokens,
            completion_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=self.cache_read_tokens,
                cache_write_tokens=self.cache_write_tokens,
                cache_creation_token_details=details,
            ),
            completion_tokens_details=(
                CompletionTokensDetailsWrapper(reasoning_tokens=self.reasoning_tokens)
                if self.reasoning_tokens is not None
                else None
            ),
        )


def _catalog_models(usage: TraceUsage) -> Iterator[ModelInfoBase]:
    for provider in usage.provider_names():
        try:
            info: Final = get_model_info_helper(
                model=usage.model, custom_llm_provider=provider, allow_dynamic=False, default_token_cost=nan
            )
        except ModelNotMappedError:
            continue
        if info["key"] in litellm.model_cost:
            yield info


def _catalog_model_info(usage: TraceUsage) -> ModelInfoBase | None:
    matches: Final = {info["key"]: info for info in _catalog_models(usage)}
    return next(iter(matches.values())) if len(matches) == 1 else None


def estimate_cost(attributes: Mapping[str, str]) -> float | None:
    try:
        usage: Final = TraceUsage.model_validate(attributes)
        model_info: Final = _catalog_model_info(usage)
        if model_info is None:
            return None
        if model_info.get("off_peak_pricing") and usage.start_time is None:
            return None
        if any(
            ("audio" in key or "image" in key or "video" in key) and value != "0" for key, value in attributes.items()
        ):
            return None
        input_cost, output_cost = generic_cost_per_token(
            model=usage.model,
            custom_llm_provider=model_info["litellm_provider"],
            usage=usage.usage(),
            service_tier=usage.service_tier,
            model_info=ModelInfo(**model_info, supported_openai_params=None),
            current_time=usage.start_time,
            strict=True,
        )
        cost: Final = input_cost + output_cost
        return cost if isfinite(cost) and cost >= 0 else None
    except Exception:
        return None


def estimate_costs(attributes: Sequence[Mapping[str, str]]) -> tuple[float | None, ...]:
    return tuple(estimate_cost(item) for item in attributes)
