"""Regenerate the Docker harness configs from the provider catalog.

Usage (from the repository root):
    scripts/sync-models               rewrite docker/opencode/opencode.json and
                                      docker/pi/models.json from providers/*.json
    scripts/sync-models --check       fail if those files are out of date
    scripts/sync-models --check-live  compare the catalog with each provider's
                                      live /models endpoint (needs API keys)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .models import DEFAULT_PROVIDER, PROVIDER_CATALOG, Provider

OPENCODE_CONFIG = Path("docker/opencode/opencode.json")
PI_MODELS = Path("docker/pi/models.json")


def render_opencode_config(existing: dict) -> dict:
    """Replace the model-related keys of the OpenCode config, keep the rest."""
    default = f"{DEFAULT_PROVIDER}/{PROVIDER_CATALOG[DEFAULT_PROVIDER].default_model}"
    providers = [p for p in PROVIDER_CATALOG.values() if "opencode" in p.harnesses]
    config = dict(existing)
    config["model"] = default
    config["small_model"] = default
    config["enabled_providers"] = [p.id for p in providers]
    config["provider"] = {
        p.id: {
            "npm": "@ai-sdk/openai-compatible",
            "name": p.name,
            "options": {"baseURL": p.base_url, "apiKey": f"{{env:{p.api_key_env}}}"},
            "models": {m.id: {"name": m.name} for m in p.models},
        }
        for p in providers
        if not p.builtin
    }
    return config


def render_pi_models() -> dict:
    return {
        "providers": {
            p.id: {
                "baseUrl": p.base_url,
                "api": "openai-completions",
                "apiKey": f"${p.api_key_env}",
                "authHeader": True,
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                },
                "models": [{"id": m.id, "name": m.name} for m in p.models],
            }
            for p in PROVIDER_CATALOG.values()
            if "pi" in p.harnesses
        }
    }


def rendered_files(root: Path) -> dict[Path, str]:
    existing = json.loads((root / OPENCODE_CONFIG).read_text(encoding="utf-8"))
    return {
        root / OPENCODE_CONFIG: _dump(render_opencode_config(existing)),
        root / PI_MODELS: _dump(render_pi_models()),
    }


def stale_files(root: Path) -> list[Path]:
    return [
        path
        for path, content in rendered_files(root).items()
        if path.read_text(encoding="utf-8") != content
    ]


def check_live(root: Path) -> int:
    """Print models added or removed upstream compared with the catalog."""
    status = 0
    for provider in PROVIDER_CATALOG.values():
        if provider.builtin:
            print(f"{provider.id}: built into OpenCode; see {provider.docs}")
            continue
        try:
            live = _fetch_model_ids(provider, root)
        except (KeyError, URLError, OSError) as exc:
            print(f"{provider.id}: could not list models ({exc})")
            status = 1
            continue
        known = set(provider.model_ids)
        missing = sorted(known - live)
        new = sorted(
            model_id
            for model_id in live - known
            if not model_id.startswith(provider.ignore)
        )
        print(f"{provider.id}: {len(live)} live models, {len(known)} in catalog")
        for model_id in missing:
            print(f"  - {model_id} (in catalog, no longer offered)")
            status = 1
        for model_id in new:
            print(f"  + {model_id} (offered, not in catalog)")
    return status


def _fetch_model_ids(provider: Provider, root: Path) -> set[str]:
    key = os.environ.get(provider.api_key_env) or _dotenv_value(
        root / ".env", provider.api_key_env
    )
    if not key:
        raise KeyError(f"{provider.api_key_env} is not set")
    request = Request(
        f"{provider.base_url}/models", headers={"Authorization": f"Bearer {key}"}
    )
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)
    return {item["id"] for item in payload.get("data", [])}


def _dotenv_value(path: Path, name: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key.strip() == name:
            return value.strip().strip("'\"") or None
    return None


def _dump(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--check-live", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()

    if args.check_live:
        return check_live(root)
    if args.check:
        stale = stale_files(root)
        for path in stale:
            print(f"out of date: {path.relative_to(root)} (run scripts/sync-models)")
        return 1 if stale else 0
    for path, content in rendered_files(root).items():
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
