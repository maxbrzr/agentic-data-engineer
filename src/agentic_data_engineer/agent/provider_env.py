from pathlib import Path


PROVIDER_ENV_REQUIREMENTS = {
    "gwdg": "SAIA_API_KEY",
    "kit": "KIT_AI_API_KEY",
}


def provider_environment_name(provider_id: str | None) -> str | None:
    return PROVIDER_ENV_REQUIREMENTS.get(provider_id or "")


def dotenv_has_value(path: Path, name: str) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        candidate = line.strip()
        if not candidate or candidate.startswith("#") or "=" not in candidate:
            continue
        key, value = candidate.removeprefix("export ").split("=", 1)
        if key.strip() == name:
            return bool(value.strip().strip("'\""))
    return False


def require_provider_environment(project_root: Path, provider_id: str | None) -> None:
    env_name = provider_environment_name(provider_id)
    if env_name is None:
        return
    env_path = project_root / ".env"
    if dotenv_has_value(env_path, env_name):
        return
    raise RuntimeError(
        f"Provider {provider_id!r} requires {env_name} in the ignored "
        f"environment file {env_path}."
    )
