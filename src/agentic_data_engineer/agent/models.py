"""Supported harnesses, providers, and models.

Providers and their models are defined in src/agentic_data_engineer/providers/
(one JSON file per provider). Edit those files when a provider changes its
models, then run `scripts/sync-models` to regenerate the Docker harness configs.
"""

import json
from dataclasses import dataclass
from importlib import resources

from ..contracts import ModelConfig


@dataclass(frozen=True, slots=True)
class Model:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class Provider:
    id: str
    name: str
    docs: str
    harnesses: tuple[str, ...]
    default_model: str
    models: tuple[Model, ...]
    base_url: str | None = None
    api_key_env: str | None = None
    builtin: bool = False
    # Live model-ID prefixes deliberately left out (external, non-chat, ...).
    ignore: tuple[str, ...] = ()

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model.id for model in self.models)


def load_providers() -> dict[str, Provider]:
    providers = {}
    files = sorted(
        (
            item
            for item in resources.files("agentic_data_engineer.providers").iterdir()
            if item.name.endswith(".json")
        ),
        key=lambda item: item.name,
    )
    for item in files:
        raw = json.loads(item.read_text(encoding="utf-8"))
        provider = Provider(
            id=item.name.removesuffix(".json"),
            name=raw["name"],
            docs=raw["docs"],
            harnesses=tuple(raw["harnesses"]),
            default_model=raw["default_model"],
            models=tuple(Model(**model) for model in raw["models"]),
            base_url=raw.get("base_url"),
            api_key_env=raw.get("api_key_env"),
            builtin=raw.get("builtin", False),
            ignore=tuple(raw.get("ignore", ())),
        )
        if provider.default_model not in provider.model_ids:
            raise ValueError(
                f"{item.name}: default_model {provider.default_model!r} is not "
                "in its models list"
            )
        providers[provider.id] = provider
    return providers


PROVIDER_CATALOG = load_providers()

HARNESSES = ("opencode", "pi")
PROVIDERS = tuple(PROVIDER_CATALOG)
PROVIDERS_BY_HARNESS = {
    harness: tuple(
        provider.id
        for provider in PROVIDER_CATALOG.values()
        if harness in provider.harnesses
    )
    for harness in HARNESSES
}
MODELS_BY_PROVIDER = {
    provider.id: provider.model_ids for provider in PROVIDER_CATALOG.values()
}
MODELS = tuple(model for models in MODELS_BY_PROVIDER.values() for model in models)

DEFAULT_HARNESS = "opencode"
DEFAULT_PROVIDER = "kit"
DEFAULT_MODEL_BY_PROVIDER = {
    provider.id: provider.default_model for provider in PROVIDER_CATALOG.values()
}


def resolve_model_config(
    harness: str, provider: str | None, model: str | None
) -> ModelConfig:
    """Fill unset provider/model from defaults and check the combination."""
    provider_id = provider or DEFAULT_PROVIDER
    if provider_id not in PROVIDERS_BY_HARNESS[harness]:
        raise ValueError(
            f"provider {provider_id!r} is not supported by harness {harness!r}; "
            f"choose from {', '.join(PROVIDERS_BY_HARNESS[harness])}"
        )
    model_id = model or DEFAULT_MODEL_BY_PROVIDER[provider_id]
    if model_id not in MODELS_BY_PROVIDER[provider_id]:
        raise ValueError(
            f"model {model_id!r} is not available from provider {provider_id!r}; "
            f"choose from {', '.join(MODELS_BY_PROVIDER[provider_id])}"
        )
    return ModelConfig(provider_id=provider_id, model_id=model_id)
