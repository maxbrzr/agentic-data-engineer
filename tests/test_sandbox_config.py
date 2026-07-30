from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SandboxConfigurationTests(unittest.TestCase):
    def test_container_runs_pinned_opencode_as_non_root(self):
        dockerfile = (
            ROOT / "docker" / "opencode-sandbox" / "Dockerfile"
        ).read_text(encoding="utf-8")

        self.assertIn("ARG OPENCODE_VERSION=1.18.9", dockerfile)
        self.assertIn('"opencode-ai@${OPENCODE_VERSION}"', dockerfile)
        self.assertIn("pandas==3.0.5", dockerfile)
        self.assertIn("scikit-learn==1.9.0", dockerfile)
        self.assertNotIn("croissant-baker", dockerfile)
        self.assertIn("USER node", dockerfile)
        self.assertIn(
            "COPY docker/opencode-sandbox/opencode.json /etc/opencode/opencode.json",
            dockerfile,
        )
        self.assertIn("OPENCODE_CONFIG=/etc/opencode/opencode.json", dockerfile)
        self.assertIn('ENTRYPOINT ["opencode-sandbox-entrypoint"]', dockerfile)

        entrypoint = (
            ROOT / "docker" / "opencode-sandbox" / "entrypoint.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("raw data mount is writable", entrypoint)
        self.assertIn("output mount is not writable", entrypoint)
        self.assertIn("agentic-data-engineer-sandbox.json", entrypoint)

    def test_compose_enforces_host_filesystem_boundary(self):
        compose = (ROOT / "compose.opencode-sandbox.yml").read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            'source: "${ADE_ROOT}"\n        target: "${ADE_ROOT}"',
            compose,
        )
        self.assertNotIn('source: "${ADE_ROOT}/.opencode', compose)
        self.assertNotIn(
            'source: "${ADE_ROOT}/docker/opencode-sandbox/opencode.json"',
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
            ROOT / "docker" / "opencode-sandbox" / "opencode.json"
        ).read_text(encoding="utf-8")
        self.assertIn('"external_directory": "allow"', container_config)
        self.assertFalse((ROOT / ".opencode").exists())

    def test_launcher_only_accepts_enabled_examples(self):
        launcher = (ROOT / "scripts" / "opencode-sandbox").read_text(
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
        self.assertNotIn("--skip-opencode-sandbox-check", cli)
        self.assertIn("sandbox_manager.ensure(key)", cli)


if __name__ == "__main__":
    unittest.main()
