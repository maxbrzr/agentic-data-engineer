import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from agentic_data_engineer.agent.opencode import OpencodeHarness, OpencodeSettings
except ImportError:
    OpencodeHarness = None
    OpencodeSettings = None
from agentic_data_engineer.config import PipelineConfig
from agentic_data_engineer.contracts import (
    AgentRequest,
    AgentRunResult,
    DatasetSpec,
    MetadataGenerationResult,
    ModelConfig,
    RetrievedDataset,
)
from agentic_data_engineer.pipeline import DataEngineeringPipeline
from agentic_data_engineer.retrieval import (
    DcatApHubRetriever,
    EXAMPLE_DATASETS,
    get_example,
    list_examples,
)


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, spec, destination_root, *, force=False):
        self.calls.append((spec, destination_root, force))
        data_dir = destination_root / spec.key
        data_dir.mkdir(parents=True, exist_ok=True)
        raw_file = data_dir / "raw.csv"
        raw_file.write_text("feature,target\n1,0\n", encoding="utf-8")
        return RetrievedDataset(
            spec=spec,
            data_dir=data_dir,
            metadata_title=spec.title,
            files=(raw_file,),
        )


class FakeHarness:
    def __init__(self):
        self.calls = []

    @property
    def name(self):
        return "fake"

    def run(self, request, model):
        self.calls.append((request, model))
        artifact = request.output_dir / "parser.py"
        artifact.write_text("# generated\n", encoding="utf-8")
        train = request.output_dir / "train.csv"
        test = request.output_dir / "test.csv"
        train.write_text("feature,target\n1,0\n", encoding="utf-8")
        test.write_text("feature,target\n2,1\n", encoding="utf-8")
        return AgentRunResult(
            harness=self.name,
            run_id="fake-run",
            output_dir=request.output_dir,
            artifacts=(artifact, train, test),
        )


class FakeMetadataGenerator:
    def __init__(self):
        self.calls = []
        self.preflight_calls = []

    @property
    def name(self):
        return "fake-metadata"

    def preflight(self, dataset):
        self.preflight_calls.append(dataset)

    def generate(self, dataset, output_dir):
        self.calls.append((dataset, output_dir))
        self.assert_agent_outputs_exist(output_dir)
        path = output_dir / "croissant.json"
        path.write_text("{}\n", encoding="utf-8")
        return MetadataGenerationResult(generator=self.name, path=path)

    @staticmethod
    def assert_agent_outputs_exist(output_dir):
        for name in ("parser.py", "train.csv", "test.csv"):
            if not (output_dir / name).is_file():
                raise AssertionError(f"Metadata ran before the agent created {name}")


class CatalogAndRetrievalTests(unittest.TestCase):
    def test_catalog_contains_exactly_three_examples(self):
        examples = list_examples()
        self.assertEqual(3, len(examples))
        self.assertEqual(set(EXAMPLE_DATASETS), {example.key for example in examples})
        self.assertEqual(examples[0], get_example(examples[0].key))

    def test_unknown_example_fails(self):
        with self.assertRaisesRegex(KeyError, "Unknown example"):
            get_example("not-enabled")

    def test_dcat_adapter_downloads_to_stable_example_directory(self):
        calls = []

        class FakeDataset:
            title = "Metadata title"

            def download(self, *, data_dir, force, verbose):
                calls.append((data_dir, force, verbose))
                data_dir.mkdir(parents=True, exist_ok=True)
                file_path = data_dir / "table.csv"
                file_path.write_text("x\n1\n", encoding="utf-8")
                asset = SimpleNamespace(path=file_path)
                return FakeCollection(data_dir, [asset])

        class FakeCollection:
            def __init__(self, root, assets):
                self.root = root
                self.assets = assets

            def __iter__(self):
                return iter(self.assets)

        factory_urls = []

        def factory(url):
            factory_urls.append(url)
            return FakeDataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            spec = get_example("chemical-process-safety")
            result = DcatApHubRetriever(
                dataset_factory=factory,
                verbose=False,
            ).retrieve(spec, root, force=True)

            self.assertEqual([spec.url], factory_urls)
            self.assertEqual(root / spec.key, result.data_dir)
            self.assertEqual("Metadata title", result.metadata_title)
            self.assertEqual((root / spec.key / "table.csv",), result.files)
            self.assertEqual([(root / spec.key, True, False)], calls)

    def test_dcat_adapter_rejects_non_catalog_dataset(self):
        unsupported = DatasetSpec(
            key="unsupported",
            title="Unsupported",
            url="https://example.invalid/dataset",
        )
        retriever = DcatApHubRetriever(dataset_factory=lambda _url: None)
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "three enabled examples"):
                retriever.retrieve(unsupported, Path(temp_dir))


class PipelineArchitectureTests(unittest.TestCase):
    def test_core_package_import_does_not_load_opencode(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = "src"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import agentic_data_engineer; "
                    "assert 'opencode_ai' not in sys.modules; "
                    "assert 'dcat_ap_hub' not in sys.modules; "
                    "assert 'croissant_baker' not in sys.modules"
                ),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_pipeline_injects_retriever_harness_and_model(self):
        retriever = FakeRetriever()
        harness = FakeHarness()
        metadata_generator = FakeMetadataGenerator()
        model = ModelConfig("provider-a", "model-a", {"temperature": 0})

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prompt = root / "agent.md"
            prompt.write_text("System prompt", encoding="utf-8")
            config = PipelineConfig(
                workspace_root=root,
                prompt_path=prompt,
                model=model,
            )
            pipeline = DataEngineeringPipeline(
                retriever=retriever,
                harness=harness,
                metadata_generator=metadata_generator,
                config=config,
            )

            result = pipeline.run("tcm-predictive-maintenance")

            self.assertEqual("fake", result.agent.harness)
            self.assertEqual(1, len(retriever.calls))
            request, received_model = harness.calls[0]
            self.assertIs(model, received_model)
            self.assertEqual(root / "data" / result.dataset.spec.key, request.dataset.data_dir)
            self.assertEqual(
                root / "output" / result.dataset.spec.key / "runs",
                request.output_dir.parent,
            )
            self.assertIn("provider-a__model-a", request.output_dir.name)
            self.assertEqual("System prompt", request.system_prompt)
            self.assertIn(str(request.dataset.data_dir), request.task_prompt)
            self.assertIn("do not create or edit that file", request.task_prompt)
            self.assertEqual("fake-metadata", result.metadata.generator)
            self.assertTrue(result.metadata.path.is_file())
            self.assertEqual(
                [(result.dataset, request.output_dir)],
                metadata_generator.calls,
            )
            self.assertEqual(
                [result.dataset],
                metadata_generator.preflight_calls,
            )
            self.assertEqual(
                request.output_dir / "provenance.json",
                result.provenance_path,
            )
            provenance = json.loads(
                result.provenance_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                {
                    "harness": "fake",
                    "provider": "provider-a",
                    "model": "model-a",
                    "run_id": "fake-run",
                    "message_id": None,
                },
                provenance["agent_run"],
            )
            train_provenance = provenance["artifacts"]["train.csv"]
            self.assertEqual("agent", train_provenance["producer_type"])
            self.assertEqual("provider-a", train_provenance["provider"])
            self.assertEqual("model-a", train_provenance["model"])
            self.assertEqual(
                hashlib.sha256(
                    (request.output_dir / "train.csv").read_bytes()
                ).hexdigest(),
                train_provenance["sha256"],
            )
            croissant_provenance = provenance["artifacts"]["croissant.json"]
            self.assertEqual(
                "metadata_generator",
                croissant_provenance["producer_type"],
            )
            self.assertEqual(
                "fake-metadata",
                croissant_provenance["generator"],
            )
            self.assertEqual(
                "model-a",
                croissant_provenance["derived_from_agent_run"]["model"],
            )

    def test_repeated_runs_use_distinct_provider_model_output_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prompt = root / "agent.md"
            prompt.write_text("System prompt", encoding="utf-8")
            pipeline = DataEngineeringPipeline(
                retriever=FakeRetriever(),
                harness=FakeHarness(),
                metadata_generator=FakeMetadataGenerator(),
                config=PipelineConfig(
                    workspace_root=root,
                    prompt_path=prompt,
                    model=ModelConfig("gwdg", "devstral/model"),
                ),
            )

            first = pipeline.run("chemical-process-safety")
            second = pipeline.run("chemical-process-safety")

            self.assertNotEqual(first.agent.output_dir, second.agent.output_dir)
            self.assertTrue(first.provenance_path.is_file())
            self.assertTrue(second.provenance_path.is_file())
            self.assertIn(
                "gwdg__devstral-model",
                first.agent.output_dir.name,
            )

    def test_pipeline_can_process_all_enabled_examples(self):
        retriever = FakeRetriever()
        harness = FakeHarness()
        metadata_generator = FakeMetadataGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prompt = root / "agent.md"
            prompt.write_text("System prompt", encoding="utf-8")
            pipeline = DataEngineeringPipeline(
                retriever=retriever,
                harness=harness,
                metadata_generator=metadata_generator,
                config=PipelineConfig(
                    workspace_root=root,
                    prompt_path=prompt,
                    model=ModelConfig("provider", "model"),
                ),
            )

            results = pipeline.run_examples()

            self.assertEqual(3, len(results))
            self.assertEqual(set(EXAMPLE_DATASETS), {result.dataset.spec.key for result in results})
            self.assertEqual(3, len(harness.calls))
            self.assertEqual(3, len(metadata_generator.calls))

    def test_pipeline_rejects_false_harness_completion_before_metadata(self):
        class MissingOutputHarness:
            @property
            def name(self):
                return "missing-output"

            def run(self, request, _model):
                (request.output_dir / "parser.py").write_text(
                    "# incomplete\n",
                    encoding="utf-8",
                )
                return AgentRunResult(
                    harness=self.name,
                    run_id="false-success",
                    output_dir=request.output_dir,
                )

        retriever = FakeRetriever()
        metadata_generator = FakeMetadataGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prompt = root / "agent.md"
            prompt.write_text("System prompt", encoding="utf-8")
            pipeline = DataEngineeringPipeline(
                retriever=retriever,
                harness=MissingOutputHarness(),
                metadata_generator=metadata_generator,
                config=PipelineConfig(
                    workspace_root=root,
                    prompt_path=prompt,
                    model=ModelConfig("provider", "model"),
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                r"Agent run 'false-success'.*train\.csv.*test\.csv",
            ):
                pipeline.run("tcm-predictive-maintenance")
            self.assertEqual([], metadata_generator.calls)

    def test_pipeline_rejects_invalid_metadata_before_agent_run(self):
        class RejectingMetadataGenerator(FakeMetadataGenerator):
            def preflight(self, dataset):
                raise ValueError("missing Croissant source metadata")

        retriever = FakeRetriever()
        harness = FakeHarness()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prompt = root / "agent.md"
            prompt.write_text("System prompt", encoding="utf-8")
            pipeline = DataEngineeringPipeline(
                retriever=retriever,
                harness=harness,
                metadata_generator=RejectingMetadataGenerator(),
                config=PipelineConfig(
                    workspace_root=root,
                    prompt_path=prompt,
                    model=ModelConfig("provider", "model"),
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing Croissant source metadata",
            ):
                pipeline.run("chemical-process-safety")

            self.assertEqual([], harness.calls)


@unittest.skipIf(OpencodeHarness is None, "OpenCode optional dependency is not installed")
class OpencodeAdapterTests(unittest.TestCase):
    def test_adapter_translates_neutral_request_and_model(self):
        created_clients = []
        artifact_dir = None

        class FakeSessionApi:
            def __init__(self):
                self.chat_calls = []

            def create(self):
                return SimpleNamespace(id="session-1")

            def chat(self, session_id, **kwargs):
                self.chat_calls.append((session_id, kwargs))
                for name in ("parser.py", "train.csv", "test.csv"):
                    (artifact_dir / name).write_text("generated\n", encoding="utf-8")
                return SimpleNamespace(id="message-1")

            def messages(self, _session_id):
                return []

        class FakeClient:
            def __init__(self):
                self.session = FakeSessionApi()

        def client_factory(**_kwargs):
            client = FakeClient()
            created_clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            data_dir.mkdir()
            output_dir = root / "output"
            artifact_dir = output_dir
            spec = get_example("industry-5-cyber-physical-systems")
            request = AgentRequest(
                dataset=RetrievedDataset(spec, data_dir, spec.title),
                output_dir=output_dir,
                system_prompt="system",
                task_prompt="task",
            )
            model = ModelConfig(
                provider_id="provider-x",
                model_id="model-y",
                parameters={"temperature": 0},
            )
            harness = OpencodeHarness(
                OpencodeSettings(
                    stream_events=False,
                    require_sandbox_preflight=False,
                ),
                client_factory=client_factory,
            )

            result = harness.run(request, model)

            self.assertEqual("opencode", result.harness)
            self.assertEqual("session-1", result.run_id)
            self.assertEqual("message-1", result.message_id)
            session_id, chat = created_clients[0].session.chat_calls[0]
            self.assertEqual("session-1", session_id)
            self.assertEqual("provider-x", chat["provider_id"])
            self.assertEqual("model-y", chat["model_id"])
            self.assertEqual({"temperature": 0}, chat["extra_body"])
            self.assertTrue((output_dir / "opencode_report_session-1.md").is_file())

    def test_adapter_continues_idle_session_until_outputs_exist(self):
        created_clients = []
        artifact_dir = None

        class FakeSessionApi:
            def __init__(self):
                self.chat_calls = []

            def create(self):
                return SimpleNamespace(id="session-continuation")

            def chat(self, session_id, **kwargs):
                self.chat_calls.append((session_id, kwargs))
                if len(self.chat_calls) == 2:
                    for name in ("parser.py", "train.csv", "test.csv"):
                        (artifact_dir / name).write_text(
                            "generated\n",
                            encoding="utf-8",
                        )
                return SimpleNamespace(id=f"message-{len(self.chat_calls)}")

            def messages(self, _session_id):
                return []

        class FakeClient:
            def __init__(self):
                self.session = FakeSessionApi()

        def client_factory(**_kwargs):
            client = FakeClient()
            created_clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            data_dir.mkdir()
            artifact_dir = root / "output"
            spec = get_example("tcm-predictive-maintenance")
            request = AgentRequest(
                dataset=RetrievedDataset(spec, data_dir, spec.title),
                output_dir=artifact_dir,
                system_prompt="system",
                task_prompt="initial task",
            )
            harness = OpencodeHarness(
                OpencodeSettings(
                    stream_events=False,
                    max_continuations=2,
                    max_provider_retries=0,
                    require_sandbox_preflight=False,
                ),
                client_factory=client_factory,
            )

            result = harness.run(request, ModelConfig("provider", "model"))

            calls = created_clients[0].session.chat_calls
            self.assertEqual(2, len(calls))
            continuation = calls[1][1]["parts"][0]["text"]
            self.assertIn("became idle before completing", continuation)
            self.assertIn("do not read complete raw tables", continuation)
            self.assertEqual("message-2", result.message_id)

    def test_adapter_stops_after_continuation_budget(self):
        created_clients = []

        class FakeSessionApi:
            def __init__(self):
                self.chat_calls = []

            def create(self):
                return SimpleNamespace(id="session-stuck")

            def chat(self, session_id, **kwargs):
                self.chat_calls.append((session_id, kwargs))
                return SimpleNamespace(id="incomplete")

            def messages(self, _session_id):
                return []

        class FakeClient:
            def __init__(self):
                self.session = FakeSessionApi()

        def client_factory(**_kwargs):
            client = FakeClient()
            created_clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            data_dir.mkdir()
            output_dir = root / "output"
            spec = get_example("tcm-predictive-maintenance")
            request = AgentRequest(
                dataset=RetrievedDataset(spec, data_dir, spec.title),
                output_dir=output_dir,
                system_prompt="system",
                task_prompt="task",
            )
            harness = OpencodeHarness(
                OpencodeSettings(
                    stream_events=False,
                    max_continuations=1,
                    max_provider_retries=0,
                    require_sandbox_preflight=False,
                ),
                client_factory=client_factory,
            )

            with self.assertRaisesRegex(
                ValueError,
                r"exhausted 1 continuation attempts",
            ):
                harness.run(request, ModelConfig("provider", "model"))

            self.assertEqual(2, len(created_clients[0].session.chat_calls))
            self.assertTrue(
                (output_dir / "opencode_report_session-stuck.md").is_file()
            )

    def test_adapter_retries_no_provider_available(self):
        created_clients = []
        sleep_calls = []
        artifact_dir = None

        class FakeSessionApi:
            def __init__(self):
                self.chat_calls = []

            def create(self):
                return SimpleNamespace(id="session-retry")

            def chat(self, session_id, **kwargs):
                self.chat_calls.append((session_id, kwargs))
                if len(self.chat_calls) == 1:
                    return SimpleNamespace(id=None)
                for name in ("parser.py", "train.csv", "test.csv"):
                    (artifact_dir / name).write_text("generated\n", encoding="utf-8")
                return SimpleNamespace(id="retried-message")

            def messages(self, _session_id):
                error = (
                    SimpleNamespace(
                        data=SimpleNamespace(message="No provider available")
                    )
                    if len(self.chat_calls) == 1
                    else None
                )
                return [
                    SimpleNamespace(
                        info=SimpleNamespace(role="assistant", error=error)
                    )
                ]

        class FakeClient:
            def __init__(self):
                self.session = FakeSessionApi()

        def client_factory(**_kwargs):
            client = FakeClient()
            created_clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            data_dir.mkdir()
            artifact_dir = root / "output"
            spec = get_example("tcm-predictive-maintenance")
            request = AgentRequest(
                dataset=RetrievedDataset(spec, data_dir, spec.title),
                output_dir=artifact_dir,
                system_prompt="system",
                task_prompt="task",
            )
            harness = OpencodeHarness(
                OpencodeSettings(
                    stream_events=False,
                    max_continuations=0,
                    max_provider_retries=1,
                    retry_backoff_seconds=0.25,
                    require_sandbox_preflight=False,
                ),
                client_factory=client_factory,
                sleeper=sleep_calls.append,
            )

            result = harness.run(request, ModelConfig("provider", "model"))

            self.assertEqual(2, len(created_clients[0].session.chat_calls))
            self.assertEqual([0.25], sleep_calls)
            self.assertEqual("retried-message", result.message_id)

    def test_adapter_preflight_accepts_matching_writable_output_mount(self):
        artifact_dir = None
        data_root = None
        attestation_reads = []

        def read_attestation(url, _expected_output):
            attestation_reads.append(url)
            return {
                "example_key": "chemical-process-safety",
                "data_root": str(data_root),
                "output_dir": str(artifact_dir),
                "data_read_only": True,
                "output_writable": True,
            }

        class FakeSessionApi:
            def create(self):
                return SimpleNamespace(id="preflight-session")

            def chat(self, _session_id, **_kwargs):
                for name in ("parser.py", "train.csv", "test.csv"):
                    (artifact_dir / name).write_text("generated\n", encoding="utf-8")
                return SimpleNamespace(id="message")

            def messages(self, _session_id):
                return []

        class FakeClient:
            def __init__(self):
                self.session = FakeSessionApi()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_root = root / "data" / "chemical-process-safety"
            data_dir = data_root / "downloaded-package"
            data_dir.mkdir(parents=True)
            artifact_dir = root / "output" / "chemical-process-safety"
            spec = get_example("chemical-process-safety")
            request = AgentRequest(
                dataset=RetrievedDataset(spec, data_dir, spec.title),
                output_dir=artifact_dir,
                system_prompt="system",
                task_prompt="task",
            )
            harness = OpencodeHarness(
                OpencodeSettings(stream_events=False),
                client_factory=lambda **_kwargs: FakeClient(),
                attestation_reader=read_attestation,
            )

            result = harness.run(request, ModelConfig("provider", "model"))

            self.assertEqual("preflight-session", result.run_id)
            self.assertEqual(
                [
                    "http://127.0.0.1:54322/"
                    "agentic-data-engineer-sandbox.json"
                ],
                attestation_reads,
            )

    def test_adapter_preflight_rejects_container_for_another_example(self):
        marker = {}
        session_creates = []

        class FakeSessionApi:
            def create(self):
                session_creates.append(True)
                return SimpleNamespace(id="must-not-be-created")

        class FakeClient:
            def __init__(self):
                self.session = FakeSessionApi()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            marker.update(
                {
                    "example_key": "tcm-predictive-maintenance",
                    "data_root": str(
                        root / "data" / "tcm-predictive-maintenance"
                    ),
                    "output_dir": str(
                        root / "output" / "tcm-predictive-maintenance"
                    ),
                    "data_read_only": True,
                    "output_writable": True,
                }
            )
            data_dir = root / "data" / "chemical-process-safety"
            data_dir.mkdir(parents=True)
            output_dir = root / "output" / "chemical-process-safety"
            spec = get_example("chemical-process-safety")
            request = AgentRequest(
                dataset=RetrievedDataset(spec, data_dir, spec.title),
                output_dir=output_dir,
                system_prompt="system",
                task_prompt="task",
            )
            harness = OpencodeHarness(
                OpencodeSettings(stream_events=False),
                client_factory=lambda **_kwargs: FakeClient(),
                attestation_reader=lambda _url, _expected_output: marker,
            )

            with self.assertRaisesRegex(
                ValueError,
                r"container example is 'tcm-predictive-maintenance'.*"
                r"start chemical-process-safety",
            ):
                harness.run(request, ModelConfig("provider", "model"))

            self.assertEqual([], session_creates)


if __name__ == "__main__":
    unittest.main()
