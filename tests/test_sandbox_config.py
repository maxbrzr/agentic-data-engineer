from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GWDG_MODELS = {
    "apertus-70b-instruct-2509",
    "deepseek-v4-flash",
    "devstral-2-123b-instruct-2512",
    "gemma-4-31b-it",
    "glm-4.7",
    "meta-llama-3.1-8b-instruct",
    "mistral-medium-3.5-128b",
    "openai-gpt-oss-120b",
    "qwen3-30b-a3b-instruct-2507",
    "qwen3-coder-next",
    "qwen3-omni-30b-a3b-instruct",
    "qwen3.5-122b-a10b",
    "qwen3.5-397b-a17b",
    "qwen3.6-27b",
    "qwen3.6-35b-a3b",
}


class SandboxConfigurationTests(unittest.TestCase):
    def test_container_runs_pinned_opencode_as_non_root(self):
        dockerfile = (
            ROOT / "docker" / "opencode" / "Dockerfile"
        ).read_text(encoding="utf-8")

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

        entrypoint = (
            ROOT / "docker" / "opencode" / "entrypoint.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("raw data mount is writable", entrypoint)
        self.assertIn("output mount is not writable", entrypoint)
        self.assertIn("agentic-data-engineer-sandbox.json", entrypoint)

    def test_compose_enforces_host_filesystem_boundary(self):
        compose = (ROOT / "compose.opencode.yml").read_text(
            encoding="utf-8"
        )

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
            'source: "${ADE_ROOT}/data/${EXAMPLE_KEY:?Set EXAMPLE_KEY}"',
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

        container_config = (
            ROOT / "docker" / "opencode" / "opencode.json"
        ).read_text(encoding="utf-8")
        parsed_config = json.loads(container_config)
        self.assertIn('"external_directory": "allow"', container_config)
        self.assertIn("gwdg", parsed_config["enabled_providers"])
        gwdg = parsed_config["provider"]["gwdg"]
        self.assertEqual("@ai-sdk/openai-compatible", gwdg["npm"])
        self.assertEqual(
            "https://chat-ai.academiccloud.de/v1",
            gwdg["options"]["baseURL"],
        )
        self.assertEqual("{env:SAIA_API_KEY}", gwdg["options"]["apiKey"])
        self.assertEqual(EXPECTED_GWDG_MODELS, set(gwdg["models"]))
        self.assertNotRegex(container_config, r"Bearer\s+[A-Za-z0-9]")
        self.assertIn("path: .env", compose)
        self.assertNotIn(".env.opencode", compose)
        self.assertFalse((ROOT / ".opencode").exists())

    def test_launcher_only_accepts_enabled_examples(self):
        launcher = (ROOT / "scripts" / "opencode").read_text(
            encoding="utf-8"
        )

        for example_key in (
            "tcm-predictive-maintenance",
            "chemical-process-safety",
            "industry-5-cyber-physical-systems",
        ):
            self.assertIn(example_key, launcher)
        self.assertIn('case "$example_key" in', launcher)
        self.assertIn('ADE_UID=$(id -u)', launcher)
        self.assertIn('ADE_GID=$(id -g)', launcher)
        self.assertIn("--force-recreate", launcher)

    def test_cli_documents_automatic_sandbox_management(self):
        cli = (
            ROOT / "src" / "agentic_data_engineer" / "cli.py"
        ).read_text(encoding="utf-8")

        self.assertIn("--no-manage-opencode-sandbox", cli)
        self.assertIn("--pi-stall-timeout", cli)
        self.assertNotIn("--skip-opencode-sandbox-check", cli)
        self.assertIn(
            "sandbox_manager.ensure(key, required_provider=provider_id)",
            cli,
        )

    def test_pi_container_and_gwdg_models_are_restricted(self):
        dockerfile = (ROOT / "docker" / "pi" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        compose = (ROOT / "compose.pi.yml").read_text(encoding="utf-8")
        entrypoint = (ROOT / "docker" / "pi" / "entrypoint.sh").read_text(
            encoding="utf-8"
        )
        runner = (ROOT / "docker" / "pi" / "runner.py").read_text(
            encoding="utf-8"
        )
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
            'source: "${ADE_ROOT}/data/${EXAMPLE_KEY:?Set EXAMPLE_KEY}"',
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

    def test_dotenv_template_is_tracked_but_secret_file_is_ignored(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        template = (ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertIn(".env", gitignore)
        self.assertIn("!.env.example", gitignore)
        self.assertEqual("SAIA_API_KEY=", template.splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
