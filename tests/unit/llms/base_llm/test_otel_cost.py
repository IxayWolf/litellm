from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Final

import pytest
import respx

import litellm
from litellm.integrations.otel.mappers.genai import GenAIMapper
from litellm.integrations.otel.model.payloads import LLMCallSpanData, LLMRequestParams, LLMUsage, RequestIdentity
from litellm.integrations.otel.model.semconv import GenAIOperation, resolve_provider
from litellm.llms.base_llm.otel_cost import estimate_cost, estimate_costs


@pytest.fixture
def usage(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mapping[str, str]]:
    monkeypatch.setitem(
        litellm.model_cost,
        "anthropic/trace-estimate-test",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
            "cache_read_input_token_cost": 0.1,
            "cache_creation_input_token_cost": 1.25,
            "cache_creation_input_token_cost_above_1hr": 2.0,
            "input_cost_per_token_priority": 3.0,
            "output_cost_per_token_priority": 4.0,
        },
    )
    yield {
        "gen_ai.response.model": "trace-estimate-test",
        "gen_ai.provider.name": "anthropic",
        "gen_ai.usage.input_tokens": "100",
        "gen_ai.usage.output_tokens": "10",
        "litellm.trace.start_ns": "0",
        "gen_ai.usage.cache_read.input_tokens": "60",
        "gen_ai.usage.cache_write.input_tokens": "20",
        "anthropic.usage.cache_creation.ephemeral_5m_input_tokens": "10",
        "anthropic.usage.cache_creation.ephemeral_1h_input_tokens": "10",
    }


def test_standard_cache_usage_uses_inclusive_input_and_existing_duration_pricing(usage: Mapping[str, str]) -> None:
    assert estimate_cost(usage) == pytest.approx(20 + 60 * 0.1 + 10 * 1.25 + 10 * 2 + 10 * 2)
    assert estimate_cost({**usage, "gen_ai.request.model": "unresolved-alias"}) == estimate_cost(usage)


@pytest.mark.parametrize(
    "field,value",
    (
        ("gen_ai.usage.input_tokens", ""),
        ("gen_ai.usage.output_tokens", "-1"),
        ("gen_ai.usage.output_tokens", "1.5"),
        ("gen_ai.usage.input_tokens", "79"),
        ("gen_ai.usage.cache_read.input_tokens", "bad"),
        ("anthropic.usage.cache_creation.ephemeral_1h_input_tokens", "11"),
        ("gen_ai.response.model", "unknown-trace-model"),
        ("gen_ai.usage.input_tokens.audio", "2"),
    ),
)
def test_invalid_or_unsupported_usage_remains_unknown(usage: Mapping[str, str], field: str, value: str) -> None:
    assert estimate_cost({**usage, field: value}) is None


def test_missing_usage_is_unknown_but_explicit_zero_is_priced(usage: Mapping[str, str]) -> None:
    identity: Final = {key: value for key, value in usage.items() if key in ("gen_ai.response.model", "gen_ai.provider.name")}
    assert estimate_cost(identity) is None
    assert estimate_cost({**identity, "gen_ai.usage.input_tokens": "0", "gen_ai.usage.output_tokens": "0"}) == 0


def test_served_tier_wins_over_requested_tier_and_invalid_span_does_not_lose_batch(usage: Mapping[str, str]) -> None:
    uncached: Final = {
        "gen_ai.response.model": usage["gen_ai.response.model"],
        "gen_ai.provider.name": "anthropic",
        "gen_ai.usage.input_tokens": "100",
        "gen_ai.usage.output_tokens": "10",
        "anthropic.response.service_tier": "priority",
        "openai.request.service_tier": "default",
    }
    assert estimate_costs((uncached, {}, usage)) == (340, None, estimate_cost(usage))


@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize("provider", ("huggingface", "ollama", "ollama_chat", "lemonade", "deepseek"))
@pytest.mark.parametrize("prefixed", (False, True))
def test_untrusted_model_names_never_fetch_metadata(
    provider: str, prefixed: bool, respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.route().respond(200, json={"max_position_embeddings": 8192})
    model: Final = "example/repo/resolve/main/large.bin#"
    attributes: Final = {
        "gen_ai.response.model": f"{provider}/{model}" if prefixed else model,
        "gen_ai.usage.input_tokens": "12",
        "gen_ai.usage.output_tokens": "3",
        **({} if prefixed else {"gen_ai.provider.name": provider}),
    }
    assert estimate_costs((attributes,)) == (None,)
    assert upstream.call_count == 0


@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize("provider", ("huggingface", "ollama", "ollama_chat", "lemonade", "openai"))
@pytest.mark.parametrize("rate", (None, 0.0, 0.5))
def test_static_catalog_requires_declared_rates_without_dynamic_lookup(
    monkeypatch: pytest.MonkeyPatch, provider: str, rate: float | None, respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.route().respond(200, json={"max_position_embeddings": 8192})
    model: Final = f"{provider}/otel-catalog-test"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"litellm_provider": provider, "mode": "chat", "input_cost_per_token": rate, "output_cost_per_token": 2.0},
    )
    actual: Final = estimate_cost(
        {"gen_ai.response.model": model, "gen_ai.usage.input_tokens": "10", "gen_ai.usage.output_tokens": "2"}
    )
    assert actual == (None if rate is None else 10 * rate + 4)
    assert upstream.call_count == 0


@pytest.mark.parametrize(
    "entry,input_tokens,service_tier,expected",
    (
        pytest.param(
            {"input_cost_per_token": "1e0", "output_cost_per_token": "2"}, 100, None, 120, id="numeric_strings"
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1, "output_cost_per_token": 2}]},
            100,
            None,
            120,
            id="tier_only_at_boundary",
        ),
        pytest.param(
            {
                "tiered_pricing": [
                    {"range": [0, 100], "input_cost_per_token": 1, "output_cost_per_token": 2},
                    {"range": [100, 200], "input_cost_per_token": 3, "output_cost_per_token": 4},
                ]
            },
            101,
            None,
            343,
            id="second_tier",
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1, "output_cost_per_token": 2}]},
            101,
            None,
            121,
            id="above_last_tier",
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1}], "output_cost_per_token": 2},
            100,
            None,
            120,
            id="tier_uses_flat_output",
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100]}], "input_cost_per_token": 1, "output_cost_per_token": 2},
            100,
            None,
            120,
            id="unpriced_tier_uses_flat",
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1}]},
            100,
            None,
            None,
            id="missing_tier_output",
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1, "output_cost_per_token": 2}]},
            0,
            None,
            None,
            id="no_tier_for_zero_input",
        ),
        pytest.param({"tiered_pricing": []}, 100, None, None, id="empty_tiers"),
        pytest.param(
            {"input_cost_per_token_priority": 1, "output_cost_per_token_priority": 2},
            100,
            "priority",
            120,
            id="service_tier_only",
        ),
        pytest.param(
            {"input_cost_per_token_priority": 1, "output_cost_per_token_priority": 2},
            100,
            None,
            None,
            id="service_tier_not_selected",
        ),
        pytest.param(
            {"input_cost_per_token_above_100_tokens": 1, "output_cost_per_token_above_100_tokens": 2},
            101,
            None,
            121,
            id="threshold_only",
        ),
        pytest.param(
            {"input_cost_per_token_above_100_tokens": 1, "output_cost_per_token_above_100_tokens": 2},
            100,
            None,
            None,
            id="threshold_not_crossed",
        ),
        pytest.param(
            {"off_peak_pricing": {"hours_utc": "00:00-00:00", "input_cost_per_token": 1, "output_cost_per_token": 2}},
            100,
            None,
            120,
            id="active_off_peak_only",
        ),
        pytest.param(
            {"off_peak_pricing": {"hours_utc": [], "input_cost_per_token": 1, "output_cost_per_token": 2}},
            100,
            None,
            None,
            id="inactive_off_peak_only",
        ),
        pytest.param(
            {"input_cost_per_character": 1, "output_cost_per_token": 2}, 100, None, None, id="non_token_input"
        ),
    ),
)
def test_estimates_use_selected_rates_instead_of_requiring_flat_prices(
    monkeypatch: pytest.MonkeyPatch,
    entry: Mapping[str, object],
    input_tokens: int,
    service_tier: str | None,
    expected: float | None,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost, "openai/otel-rate-test", {"litellm_provider": "openai", "mode": "chat", **entry}
    )
    attributes: Final = {
        "gen_ai.response.model": "openai/otel-rate-test",
        "litellm.trace.start_ns": "0",
        "gen_ai.usage.input_tokens": str(input_tokens),
        "gen_ai.usage.output_tokens": "10",
        **({"openai.response.service_tier": service_tier} if service_tier is not None else {}),
    }
    assert estimate_cost(attributes) == expected


@pytest.mark.parametrize("invalid_rate", ("bad", -1.0, float("nan"), float("inf"), True, False))
@pytest.mark.parametrize("representation", ("flat", "tier", "priority", "threshold", "off_peak"))
def test_invalid_selected_rates_never_become_free_or_offset_another_charge(
    monkeypatch: pytest.MonkeyPatch, invalid_rate: str | float, representation: str
) -> None:
    rates: Final = {"input_cost_per_token": invalid_rate, "output_cost_per_token": 100}
    entries: Final = {
        "flat": rates,
        "tier": {"tiered_pricing": [{"range": [0, 1000], **rates}]},
        "priority": {"input_cost_per_token_priority": invalid_rate, "output_cost_per_token_priority": 100},
        "threshold": {
            "input_cost_per_token_above_10_tokens": invalid_rate,
            "output_cost_per_token_above_10_tokens": 100,
        },
        "off_peak": {
            "input_cost_per_token": 1,
            "output_cost_per_token": 100,
            "off_peak_pricing": {"hours_utc": "00:00-00:00", **rates},
        },
    }
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/otel-invalid-test",
        {"litellm_provider": "openai", "mode": "chat", **entries[representation]},
    )
    assert (
        estimate_cost(
            {
                "gen_ai.response.model": "openai/otel-invalid-test",
                "litellm.trace.start_ns": "0",
                "gen_ai.usage.input_tokens": "100",
                "gen_ai.usage.output_tokens": "10",
                "openai.response.service_tier": "priority",
            }
        )
        is None
    )


@pytest.mark.parametrize(
    "cache_rates,expected",
    (
        ({}, 120),
        (
            {
                "cache_read_input_token_cost": 0,
                "cache_creation_input_token_cost": 0,
                "cache_creation_input_token_cost_above_1hr": 0,
            },
            40,
        ),
        ({"cache_read_input_token_cost": "bad"}, None),
    ),
)
def test_tier_cache_rates_preserve_defaults_explicit_zero_and_invalid_values(
    monkeypatch: pytest.MonkeyPatch, usage: Mapping[str, str], cache_rates: Mapping[str, object], expected: float | None
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "anthropic/trace-estimate-test",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "tiered_pricing": [
                {"range": [0, 1000], "input_cost_per_token": 1, "output_cost_per_token": 2, **cache_rates}
            ],
        },
    )
    assert estimate_cost(usage) == expected


@pytest.mark.parametrize("tiered", (False, True))
@pytest.mark.parametrize(
    "write_rate,write_1h_rate,tokens_5m,tokens_1h,expected",
    (
        ("bad", 2, "0", "20", 140),
        (2, "bad", "20", "0", 140),
        (2, "bad", None, None, 140),
        ("bad", 2, "20", "0", None),
        (2, "bad", "0", "20", None),
        ("bad", 2, None, None, None),
    ),
)
def test_cache_write_rate_validation_follows_used_duration(
    monkeypatch: pytest.MonkeyPatch,
    usage: Mapping[str, str],
    tiered: bool,
    write_rate: str | int,
    write_1h_rate: str | int,
    tokens_5m: str | None,
    tokens_1h: str | None,
    expected: float | None,
) -> None:
    rates: Final = {
        "input_cost_per_token": 1,
        "output_cost_per_token": 2,
        "cache_creation_input_token_cost": write_rate,
        "cache_creation_input_token_cost_above_1hr": write_1h_rate,
    }
    monkeypatch.setitem(
        litellm.model_cost,
        "anthropic/trace-estimate-test",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            **({"tiered_pricing": [{"range": [0, 1000], **rates}]} if tiered else rates),
        },
    )
    attributes: Final = {key: value for key, value in usage.items() if "ephemeral_" not in key}
    if tokens_5m is not None and tokens_1h is not None:
        attributes["anthropic.usage.cache_creation.ephemeral_5m_input_tokens"] = tokens_5m
        attributes["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"] = tokens_1h
    assert estimate_cost(attributes) == expected


def test_unused_cache_rates_cannot_contaminate_uncached_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/otel-uncached-test",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "cache_read_input_token_cost": "bad",
            "cache_creation_input_token_cost": "bad",
            "cache_creation_input_token_cost_above_1hr": "bad",
        },
    )
    assert estimate_cost(
        {
            "gen_ai.response.model": "openai/otel-uncached-test",
            "gen_ai.usage.input_tokens": "100",
            "gen_ai.usage.output_tokens": "10",
        }
    ) == 120


@pytest.mark.parametrize(
    "aliases",
    (
        {"gen_ai.usage.cache_write.input_tokens": "gen_ai.usage.cache_creation.input_tokens"},
        {
            "gen_ai.usage.input_tokens": "gen_ai.usage.prompt_tokens",
            "gen_ai.usage.output_tokens": "gen_ai.usage.completion_tokens",
        },
        {
            "gen_ai.provider.name": "gen_ai.system",
            "gen_ai.usage.input_tokens": "gen_ai.usage.prompt_tokens",
            "gen_ai.usage.output_tokens": "gen_ai.usage.completion_tokens",
            "gen_ai.usage.cache_write.input_tokens": "gen_ai.usage.cache_creation.input_tokens",
        },
    ),
)
def test_legacy_genai_usage_keeps_inclusive_cache_math(usage: Mapping[str, str], aliases: Mapping[str, str]) -> None:
    legacy: Final = {aliases.get(key, key): value for key, value in usage.items()}
    assert estimate_cost(legacy) == pytest.approx(78.5)
    assert estimate_cost({**usage, **legacy}) == pytest.approx(78.5)
    assert estimate_cost({**usage, **legacy, "gen_ai.usage.cache_creation.input_tokens": "020"}) == pytest.approx(78.5)


@pytest.mark.parametrize(
    "alias",
    ("gen_ai.usage.prompt_tokens", "gen_ai.usage.completion_tokens", "gen_ai.usage.cache_creation.input_tokens"),
)
@pytest.mark.parametrize("value", ("1", "bad", "-1", "18446744073709551616"))
def test_invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key(
    usage: Mapping[str, str], alias: str, value: str
) -> None:
    assert estimate_cost({**usage, alias: value}) is None


def test_legacy_cache_write_never_adds_to_an_inclusive_input_total(usage: Mapping[str, str]) -> None:
    legacy: Final = {key: value for key, value in usage.items() if key != "gen_ai.usage.cache_write.input_tokens"}
    assert (
        estimate_cost({**legacy, "gen_ai.usage.cache_creation.input_tokens": "20", "gen_ai.usage.input_tokens": "79"})
        is None
    )


@pytest.mark.parametrize(
    "tiers,expected",
    (
        ({"gen_ai.openai.response.service_tier": "priority"}, 340),
        ({"gen_ai.openai.request.service_tier": "priority"}, 340),
        ({"gen_ai.openai.response.service_tier": "priority", "openai.request.service_tier": "default"}, 340),
        ({"openai.response.service_tier": "default", "gen_ai.openai.request.service_tier": "priority"}, 120),
        ({"openai.response.service_tier": "priority", "gen_ai.openai.response.service_tier": "priority"}, 340),
        ({"openai.response.service_tier": "priority", "gen_ai.openai.response.service_tier": "default"}, None),
        ({"anthropic.response.service_tier": "priority", "gen_ai.openai.response.service_tier": "default"}, None),
        ({"openai.request.service_tier": "priority", "gen_ai.openai.request.service_tier": "default"}, None),
    ),
)
def test_tier_aliases_preserve_served_precedence_and_reject_same_fact_conflicts(
    usage: Mapping[str, str], tiers: Mapping[str, str], expected: float | None
) -> None:
    attributes: Final = {key: value for key, value in usage.items() if "cache" not in key}
    assert estimate_cost({**attributes, **tiers}) == expected


def test_actual_genai_mapper_cache_creation_is_priced(usage: Mapping[str, str]) -> None:
    data: Final = LLMCallSpanData(
        operation=GenAIOperation.CHAT,
        provider="anthropic",
        request_model=usage["gen_ai.response.model"],
        response_model=None,
        response_id=None,
        request_params=LLMRequestParams(),
        usage=LLMUsage(input_tokens=100, output_tokens=10, cache_read_input_tokens=60, cache_creation_input_tokens=20),
        finish_reasons=(),
        error=None,
        response_cost=None,
        server=None,
        identity=RequestIdentity(),
    )
    attributes: Final = {key: str(value) for key, value in GenAIMapper().map(data).items()}
    assert estimate_cost(attributes) == 71


@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize(
    "provider",
    (
        "azure",
        "azure_ai",
        "gemini",
        "mistral",
        "xai",
        "watsonx",
        "cohere_chat",
        "text-completion-openai",
        "bedrock",
        "vertex_ai",
    ),
)
def test_canonical_provider_values_reuse_emitter_mapping_without_network(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, provider: str
) -> None:
    upstream: Final = respx_mock.route().respond(200, json={})
    monkeypatch.setitem(
        litellm.model_cost,
        f"{provider}/otel-provider-test",
        {"litellm_provider": provider, "mode": "chat", "input_cost_per_token": 1, "output_cost_per_token": 2},
    )
    attributes: Final = {
        "gen_ai.response.model": "otel-provider-test",
        "gen_ai.provider.name": resolve_provider(provider),
        "gen_ai.usage.input_tokens": "100",
        "gen_ai.usage.output_tokens": "10",
    }
    assert estimate_cost(attributes) == 120
    assert estimate_cost({**attributes, "gen_ai.system": provider}) == 120
    assert upstream.call_count == 0


def test_provider_ambiguity_needs_a_unique_catalog_row_or_explicit_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    for provider, rate in (("cohere", 1), ("cohere_chat", 9)):
        monkeypatch.setitem(
            litellm.model_cost,
            f"{provider}/otel-provider-test",
            {"litellm_provider": provider, "mode": "chat", "input_cost_per_token": rate, "output_cost_per_token": 2},
        )
    attributes: Final = {
        "gen_ai.response.model": "otel-provider-test",
        "gen_ai.provider.name": "cohere",
        "gen_ai.usage.input_tokens": "100",
        "gen_ai.usage.output_tokens": "10",
    }
    assert estimate_cost(attributes) is None
    assert estimate_cost({**attributes, "gen_ai.response.model": "cohere_chat/otel-provider-test"}) == 920
    assert estimate_cost({**attributes, "gen_ai.system": "cohere_chat"}) == 920
    assert estimate_cost({**attributes, "gen_ai.system": "openai"}) is None


@pytest.mark.parametrize(
    "reasoning,rate,output,expected",
    (
        ("4", 5, "10", 132),
        ("4", 0, "10", 112),
        ("4", None, "10", 120),
        (None, 5, "10", None),
        (None, 2, "10", 120),
        (None, "bad", "10", None),
        ("0", "bad", "10", 120),
        (None, "bad", "0", 100),
        ("11", 5, "10", None),
        ("-1", 5, "10", None),
        ("bad", 5, "10", None),
        ("18446744073709551616", 5, "10", None),
    ),
)
def test_reasoning_is_an_inclusive_output_subset_with_explicit_unknowns(
    monkeypatch: pytest.MonkeyPatch,
    reasoning: str | None,
    rate: str | int | None,
    output: str,
    expected: float | None,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/otel-reasoning-test",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "output_cost_per_reasoning_token": rate,
        },
    )
    attributes: Final = {
        "gen_ai.response.model": "openai/otel-reasoning-test",
        "gen_ai.usage.input_tokens": "100",
        "gen_ai.usage.output_tokens": output,
        **({"gen_ai.usage.reasoning.output_tokens": reasoning} if reasoning is not None else {}),
    }
    assert estimate_cost(attributes) == expected


@pytest.mark.parametrize("rate", (5, "bad", -1, float("nan"), float("inf"), True, False))
@pytest.mark.parametrize("representation", ("flat", "tier", "priority", "off_peak"))
def test_reasoning_uses_strict_selected_tier_rates(
    monkeypatch: pytest.MonkeyPatch, rate: str | float, representation: str
) -> None:
    entries: Final = {
        "flat": {"output_cost_per_reasoning_token": rate},
        "tier": {
            "tiered_pricing": [
                {
                    "range": [0, 1000],
                    "input_cost_per_token": 1,
                    "output_cost_per_token": 2,
                    "output_cost_per_reasoning_token": rate,
                }
            ]
        },
        "priority": {"output_cost_per_reasoning_token_priority": rate},
        "off_peak": {"off_peak_pricing": {"hours_utc": "00:00-00:00", "output_cost_per_reasoning_token": rate}},
    }
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/otel-selected-reasoning",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            **entries[representation],
        },
    )
    attributes: Final = {
        "gen_ai.response.model": "openai/otel-selected-reasoning",
        "gen_ai.usage.input_tokens": "100",
        "gen_ai.usage.output_tokens": "10",
        "gen_ai.usage.reasoning.output_tokens": "4",
        "openai.response.service_tier": "priority",
        "litellm.trace.start_ns": "0",
    }
    assert estimate_cost(attributes) == (132 if rate == 5 else None)


def test_selected_output_rate_can_make_a_missing_reasoning_split_irrelevant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/otel-priority-reasoning",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "output_cost_per_reasoning_token": 9,
            "output_cost_per_token_priority": 4,
        },
    )
    assert (
        estimate_cost(
            {
                "gen_ai.response.model": "openai/otel-priority-reasoning",
                "gen_ai.usage.input_tokens": "100",
                "gen_ai.usage.output_tokens": "10",
                "openai.response.service_tier": "priority",
            }
        )
        == 140
    )


def test_off_peak_prices_follow_captured_time_including_nanosecond_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/otel-timed-test",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 2,
            "output_cost_per_token": 4,
            "off_peak_pricing": {"hours_utc": "04:00-08:00", "input_cost_per_token": 1, "output_cost_per_token": 2},
        },
    )
    attributes: Final = {
        "gen_ai.response.model": "openai/otel-timed-test",
        "gen_ai.usage.input_tokens": "100",
        "gen_ai.usage.output_tokens": "10",
    }
    opening: Final = int(datetime(2026, 1, 5, 4, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    closing: Final = opening + 4 * 3600 * 1_000_000_000
    assert estimate_cost({**attributes, "litellm.trace.start_ns": str(opening - 1)}) == 240
    assert estimate_cost({**attributes, "litellm.trace.start_ns": str(opening)}) == 120
    assert estimate_cost({**attributes, "litellm.trace.start_ns": str(closing - 1)}) == 120
    assert estimate_cost({**attributes, "litellm.trace.start_ns": str(closing)}) == 240
    assert estimate_cost(attributes) is None
    assert estimate_cost({**attributes, "litellm.trace.start_ns": "-1"}) == 240


@pytest.mark.parametrize("timestamp", ("bad", "9223372036854775808", "-9223372036854775809"))
def test_invalid_supplied_call_time_remains_unknown(usage: Mapping[str, str], timestamp: str) -> None:
    assert estimate_cost({**usage, "litellm.trace.start_ns": timestamp}) is None
