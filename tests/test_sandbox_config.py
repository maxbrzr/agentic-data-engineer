import json
import unittest
from pathlib import Path

from agentic_data_engineer.agent.models import (
    DEFAULT_MODEL_BY_PROVIDER,
    DEFAULT_PROVIDER,
    MODELS_BY_PROVIDER,
    resolve_model_config,
)
from agentic_data_engineer.agent.model_sync import stale_files
from agentic_data_engineer.contracts import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GWDG_MODELS = set(MODELS_BY_PROVIDER["gwdg"])
EXPECTED_KIT_MODELS = set(MODELS_BY_PROVIDER["kit"])


class SandboxConfigurationTests(unittest.TestCase):
    def test_container_runs_pinned_opencode_as_non_root(self):
        dockerfile = (ROOT / "docker" / "opencode" / "Dockerfile").read_text(
            encoding="utf-8"
        )

        self.assertIn("ARG OPENCODE_VERSION=1.18.9", dockerfile)
        self.assertIn('"opencode-ai@${OPENCODE_VERSION}"', dockerfile)
        self.assertIn("pandas==3.0.5", dockerfile)
        self.assertIn("scikit-learn==1.9.0", dockerfile)
        self.assertNotIn("croissant-baker", dockerfile)
        self.assertIn("USER node", dockerfile)
        self.assertIn(
            "COPY docker/opencode/opencode.json /etc/opencode/opencode.json",
            dockerfile,
        )
        self.assertIn("OPENCODE_CONFIG=/etc/opencode/opencode.json", dockerfile)
        self.assertIn('ENTRYPOINT ["opencode-entrypoint"]', dockerfile)

        entrypoint = (ROOT / "docker" / "opencode" / "entrypoint.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("raw data mount is writable", entrypoint)
        self.assertIn("output mount is not writable", entrypoint)
        self.assertIn("agentic-data-engineer-sandbox.json", entrypoint)

    def test_compose_enforces_host_filesystem_boundary(self):
        compose = (ROOT / "compose.opencode.yml").read_text(encoding="utf-8")

        self.assertNotIn(
            'source: "${ADE_ROOT}"\n        target: "${ADE_ROOT}"',
            compose,
        )
        self.assertNotIn('source: "${ADE_ROOT}/.opencode', compose)
        self.assertNotIn(
            'source: "${ADE_ROOT}/docker/opencode/opencode.json"',
            compose,
        )
        self.assertIn("OPENCODE_CONFIG: /etc/opencode/opencode.json", compose)
        self.assertNotIn("${ADE_ROOT}:rw,nosuid", compose)
        self.assertIn(
            'source: "${ADE_ROOT}/data/${DATA_DIR_NAME:?Set DATA_DIR_NAME}"',
            compose,
        )
        self.assertIn(
            'source: "${ADE_ROOT}/output/${EXAMPLE_KEY:?Set EXAMPLE_KEY}"',
            compose,
        )
        # One read-only data bind plus the service's read-only root filesystem.
        self.assertEqual(2, compose.count("read_only: true"))
        self.assertIn('"127.0.0.1:54321:54321"', compose)
        self.assertIn('"127.0.0.1:54322:54322"', compose)
        self.assertIn("no-new-privileges:true", compose)
        self.assertIn("cap_drop:", compose)
        self.assertIn("- ALL", compose)
        self.assertNotIn("docker.sock", compose)
        self.assertIn(
            'ADE_EXAMPLE_KEY: "${EXAMPLE_KEY:?Set EXAMPLE_KEY}"',
            compose,
        )
        self.assertIn(
            "ADE_SANDBOX_MARKER: /tmp/agentic-data-engineer-sandbox.json",
            compose,
        )
        self.assertIn(
            "54322/health",
            compose,
        )

        container_config = (ROOT / "docker" / "opencode" / "opencode.json").read_text(
            encoding="utf-8"
        )
        parsed_config = json.loads(container_config)
        self.assertIn('"external_directory": "allow"', container_config)
        self.assertIn("gwdg", parsed_config["enabled_providers"])
        self.assertIn("kit", parsed_config["enabled_providers"])
        gwdg = parsed_config["provider"]["gwdg"]
        self.assertEqual("@ai-sdk/openai-compatible", gwdg["npm"])
        self.assertEqual(
            "https://chat-ai.academiccloud.de/v1",
            gwdg["options"]["baseURL"],
        )
        self.assertEqual("{env:SAIA_API_KEY}", gwdg["options"]["apiKey"])
        self.assertEqual(EXPECTED_GWDG_MODELS, set(gwdg["models"]))
        default_model = (
            f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER]}"
        )
        self.assertEqual(default_model, parsed_config["model"])
        self.assertEqual(default_model, parsed_config["small_model"])
        kit = parsed_config["provider"]["kit"]
        self.assertEqual("@ai-sdk/openai-compatible", kit["npm"])
        self.assertEqual(
            "https://ki-toolbox.scc.kit.edu/api/v1",
            kit["options"]["baseURL"],
        )
        self.assertEqual("{env:KIT_TOOLBOX_API_KEY}", kit["options"]["apiKey"])
        self.assertEqual(EXPECTED_KIT_MODELS, set(kit["models"]))
        self.assertNotRegex(container_config, r"Bearer\s+[A-Za-z0-9]")
        self.assertIn("path: .env", compose)
        self.assertNotIn(".env.opencode", compose)
        self.assertFalse((ROOT / ".opencode").exists())

    def test_launcher_accepts_safe_dataset_keys(self):
        launcher = (ROOT / "scripts" / "opencode").read_text(encoding="utf-8")

        self.assertIn('case "$example_key" in', launcher)
        self.assertIn("*[!A-Za-z0-9._-]*", launcher)
        self.assertIn("*/*", launcher)
        self.assertIn("ADE_UID=$(id -u)", launcher)
        self.assertIn("ADE_GID=$(id -g)", launcher)
        self.assertIn("--force-recreate", launcher)

    def test_cli_documents_automatic_sandbox_management(self):
        cli = (ROOT / "src" / "agentic_data_engineer" / "cli.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("--no-manage-opencode-sandbox", cli)
        self.assertIn("--pi-stall-timeout", cli)
        self.assertIn('"--url"', cli)
        self.assertNotIn("--skip-opencode-sandbox-check", cli)
        self.assertIn(
            "data_dir_name=spec.local_dir_name",
            cli,
        )

    def test_docker_model_configs_match_provider_catalog(self):
        self.assertEqual(
            [],
            [path.name for path in stale_files(ROOT)],
            "Docker model configs are stale; run scripts/sync-models",
        )

    def test_model_defaults_and_compatibility(self):
        default = ModelConfig(
            DEFAULT_PROVIDER, DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER]
        )
        self.assertEqual(resolve_model_config("opencode", None, None), default)
        self.assertEqual(resolve_model_config("pi", None, None), default)
        self.assertEqual(
            resolve_model_config("pi", "gwdg", None),
            ModelConfig("gwdg", "qwen3-coder-next"),
        )
        self.assertEqual(
            resolve_model_config("pi", "kit", None),
            ModelConfig("kit", "kit.glm-5.3-flash"),
        )
        self.assertEqual(
            resolve_model_config("opencode", "opencode", None),
            ModelConfig("opencode", "deepseek-v4.1-flash-free"),
        )
        self.assertEqual(
            resolve_model_config("pi", "gwdg", "glm-5.3-flash"),
            ModelConfig("gwdg", "glm-5.3-flash"),
        )
        with self.assertRaisesRegex(ValueError, "not supported by harness"):
            resolve_model_config("pi", "opencode", None)
        with self.assertRaisesRegex(ValueError, "not available from provider"):
            resolve_model_config("opencode", "kit", "glm-5.3-flash")

    def test_pi_container_and_remote_models_are_restricted(self):
        dockerfile = (ROOT / "docker" / "pi" / "Dockerfile").read_text(encoding="utf-8")
        compose = (ROOT / "compose.pi.yml").read_text(encoding="utf-8")
        entrypoint = (ROOT / "docker" / "pi" / "entrypoint.sh").read_text(
            encoding="utf-8"
        )
        runner = (ROOT / "docker" / "pi" / "runner.py").read_text(encoding="utf-8")
        models = json.loads(
            (ROOT / "docker" / "pi" / "models.json").read_text(encoding="utf-8")
        )

        self.assertIn("@earendil-works/pi-coding-agent@${PI_VERSION}", dockerfile)
        self.assertIn("USER node", dockerfile)
        self.assertIn('ENTRYPOINT ["pi-entrypoint"]', dockerfile)
        self.assertIn("raw data mount is writable", entrypoint)
        self.assertIn("output mount is not writable", entrypoint)
        self.assertIn("request file is outside", entrypoint)
        self.assertIn('"read,bash,edit,write,grep,find,ls"', runner)
        self.assertIn("stdin=subprocess.DEVNULL", runner)
        self.assertIn(
            'source: "${ADE_ROOT}/data/${DATA_DIR_NAME:?Set DATA_DIR_NAME}"',
            compose,
        )
        self.assertIn(
            'source: "${ADE_ROOT}/output/${EXAMPLE_KEY:?Set EXAMPLE_KEY}"',
            compose,
        )
        self.assertEqual(2, compose.count("read_only: true"))
        self.assertIn("no-new-privileges:true", compose)
        self.assertNotIn("docker.sock", compose)
        self.assertNotIn("ports:", compose)
        gwdg = models["providers"]["gwdg"]
        self.assertEqual("openai-completions", gwdg["api"])
        self.assertEqual("$SAIA_API_KEY", gwdg["apiKey"])
        self.assertTrue(gwdg["authHeader"])
        self.assertEqual(EXPECTED_GWDG_MODELS, {item["id"] for item in gwdg["models"]})
        kit = models["providers"]["kit"]
        self.assertEqual("openai-completions", kit["api"])
        self.assertEqual("https://ki-toolbox.scc.kit.edu/api/v1", kit["baseUrl"])
        self.assertEqual("$KIT_TOOLBOX_API_KEY", kit["apiKey"])
        self.assertTrue(kit["authHeader"])
        self.assertEqual(EXPECTED_KIT_MODELS, {item["id"] for item in kit["models"]})

    def test_dotenv_template_is_tracked_but_secret_file_is_ignored(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        template = (ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertIn(".env", gitignore)
        self.assertIn("!.env.example", gitignore)
        self.assertIn("SAIA_API_KEY=", template.splitlines())
        self.assertIn("KIT_TOOLBOX_API_KEY=", template.splitlines())


if __name__ == "__main__":
    unittest.main()
