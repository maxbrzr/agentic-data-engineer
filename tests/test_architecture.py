import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agentic_data_engineer.agent.opencode import OpencodeHarness, OpencodeSettings
from agentic_data_engineer.config import PipelineConfig
from agentic_data_engineer.contracts import (
    AgentRequest,
    AgentRunResult,
    DatasetSpec,
    MetadataGenerationResult,
    ModelConfig,
    ProcessorCheckResult,
    RetrievedDataset,
)
from agentic_data_engineer.pipeline import DataEngineeringPipeline
from agentic_data_engineer.retrieval import (
    DcatApHubRetriever,
    EXAMPLE_DATASETS,
    dataset_spec_from_url,
    get_example,
    list_examples,
    validate_dataset_key,
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
        self.saw_data_copy = (request.output_dir / "data").exists()
        artifact = request.output_dir / "processor.py"
        artifact.write_text(
            "def process(data_dir, output_dir):\n    pass\n", encoding="utf-8"
        )
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
        for name in ("processor.py", "train.csv", "test.csv"):
            if not (output_dir / name).is_file():
                raise AssertionError(f"Metadata ran before the agent created {name}")


# Datasets without a local folder, so the retriever downloads them. They are not
# catalog examples; the tests only need specs for remote records.
TCM_SPEC = DatasetSpec(
    key="tcm-predictive-maintenance",
    title="TCM: Benchmark Datasets for Predictive Maintenance in Steel Manufacturing",
    url="http://data.sdil.de/en/dataset/10-5281-zenodo-11469702",
)
CHEMICAL_SPEC = DatasetSpec(
    key="chemical-process-safety",
    title="AI-Ready Simulation Dataset for Chemical Process Safety and Accident Prevention",
    url="http://data.sdil.de/en/dataset/10-5281-zenodo-18200704",
)
INDUSTRY5_SPEC = DatasetSpec(
    key="industry-5-cyber-physical-systems",
    title=(
        "Smart Cyber-Physical Systems Dataset for Real-Time Optimization, "
        "Resilience, and Sustainability in Industry 5.0"
    ),
    url="http://data.sdil.de/en/dataset/10-5281-zenodo-18177239",
)


class FakeProcessorChecker:
    """Stands in for the isolated re-run; records what it was asked to check."""

    def __init__(self):
        self.calls = []

    @property
    def name(self):
        return "fake-processor-check"

    def check(self, package_dir, expected):
        self.calls.append((package_dir, dict(expected)))
        return ProcessorCheckResult(
            checker=self.name,
            sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in expected.items()},
            runtime="fake",
        )


class CatalogAndRetrievalTests(unittest.TestCase):
    def test_catalog_contains_configured_examples(self):
        examples = list_examples()
        self.assertEqual(len(EXAMPLE_DATASETS), len(examples))
        self.assertEqual(set(EXAMPLE_DATASETS), {example.key for example in examples})
        self.assertEqual(examples[0], get_example(examples[0].key))
        self.assertEqual(
            "Predictive Maintenance Engine Data",
            get_example("predictive-maintenance-engine-data").local_dir_name,
        )

    def test_examples_are_local_folders_with_a_record(self):
        # data/ is not versioned; check the folders only where they exist.
        data_root = Path(__file__).resolve().parents[1] / "data"
        for example in list_examples():
            self.assertTrue(example.local_dir_name, example.key)
            folder = data_root / example.local_dir_name
            if data_root.is_dir():
                self.assertTrue((folder / "dcat-ap-metadata.jsonld").is_file(), folder)

    def test_unknown_example_fails(self):
        with self.assertRaisesRegex(KeyError, "Unknown example"):
            get_example("not-enabled")

    def test_url_builds_stable_safe_dataset_spec(self):
        url = "https://catalog.example.org/datasets/process-safety"

        first = dataset_spec_from_url(url)
        second = dataset_spec_from_url(f"  {url}  ")

        self.assertEqual(first, second)
        self.assertRegex(first.key, r"^url-[0-9a-f]{12}$")
        self.assertEqual(url, first.url)
        self.assertIn("catalog.example.org", first.title)

    def test_url_rejects_non_http_sources(self):
        for url in ("relative.jsonld", "file:///tmp/dataset.jsonld", ""):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, "absolute HTTP"):
                    dataset_spec_from_url(url)

    def test_dataset_key_rejects_path_traversal(self):
        for key in ("../escape", ".hidden", "nested/path", "with space"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "Dataset keys"):
                    validate_dataset_key(key)

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
            spec = CHEMICAL_SPEC
            result = DcatApHubRetriever(
                dataset_factory=factory,
                verbose=False,
            ).retrieve(spec, root, force=True)

            self.assertEqual([spec.url], factory_urls)
            self.assertEqual(root / spec.key, result.data_dir)
            self.assertEqual("Metadata title", result.metadata_title)
            self.assertEqual((root / spec.key / "table.csv",), result.files)
            self.assertEqual([(root / spec.key, True, False)], calls)

    def test_dcat_adapter_accepts_url_derived_dataset(self):
        remote = DatasetSpec(
            key="url-123456789abc",
            title="Remote",
            url="https://example.invalid/dataset",
        )

        class EmptyCollection:
            def __init__(self, root):
                self.root = root

            def __iter__(self):
                return iter(())

        class RemoteDataset:
            title = "Remote metadata title"

            def download(self, *, data_dir, force, verbose):
                data_dir.mkdir(parents=True, exist_ok=True)
                return EmptyCollection(data_dir)

        retriever = DcatApHubRetriever(dataset_factory=lambda _url: RemoteDataset())
        with tempfile.TemporaryDirectory() as temp_dir:
            result = retriever.retrieve(remote, Path(temp_dir))

        self.assertEqual(remote, result.spec)
        self.assertEqual("Remote metadata title", result.metadata_title)

    def test_dcat_adapter_rejects_html_instead_of_dataset_asset(self):
        class HtmlCollection:
            def __init__(self, root, asset):
                self.root = root
                self.asset = asset

            def __iter__(self):
                return iter((self.asset,))

        class LoginPageDataset:
            title = "Protected dataset"

            def download(self, *, data_dir, force, verbose):
                data_dir.mkdir(parents=True, exist_ok=True)
                path = data_dir / "archive.html"
                path.write_text("<html>Sign in</html>", encoding="utf-8")
                return HtmlCollection(data_dir, SimpleNamespace(path=path))

        spec = dataset_spec_from_url("https://example.test/protected")
        retriever = DcatApHubRetriever(
            dataset_factory=lambda _url: LoginPageDataset()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "authentication"):
                retriever.retrieve(spec, Path(temp_dir))

    def test_dcat_adapter_reuses_configured_local_dataset_directory(self):
        calls = []

        class LocalCollection:
            def __init__(self, root, assets):
                self.root = root
                self.assets = assets

            def __iter__(self):
                return iter(self.assets)

        class LocalDataset:
            title = "Local title"

            def download(self, *, data_dir, force, verbose):
                calls.append((data_dir, force, verbose))
                asset = SimpleNamespace(path=data_dir / "table.csv")
                return LocalCollection(data_dir, [asset])

        spec = DatasetSpec(
            key="local-example",
            title="Local Example",
            url="https://example.test/local",
            local_dir_name="Local Dataset Directory",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            local_dir = root / "Local Dataset Directory"
            local_dir.mkdir()
            (local_dir / "table.csv").write_text("x\n1\n", encoding="utf-8")
            result = DcatApHubRetriever(
                dataset_factory=lambda _url: self.fail("URL fetch was used"),
                local_dataset_factory=lambda path: (
                    calls.append(("factory", path)) or LocalDataset()
                ),
                verbose=False,
            ).retrieve(spec, root, force=True)

            self.assertEqual(local_dir, result.data_dir)
            self.assertEqual((local_dir / "table.csv",), result.files)
            self.assertEqual(("factory", local_dir), calls[0])
            self.assertEqual((local_dir, False, False), calls[1])


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
                processor_checker=FakeProcessorChecker(),
                config=config,
            )

            result = pipeline.run_spec(TCM_SPEC)

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
            self.assertFalse(harness.saw_data_copy)
            copied = request.output_dir / "data" / "raw.csv"
            self.assertEqual("feature,target\n1,0\n", copied.read_text(encoding="utf-8"))
            source_data = provenance["source_data"]
            self.assertEqual(
                str(request.dataset.data_dir.relative_to(root)), source_data["copied_from"]
            )
            self.assertEqual(request.dataset.spec.url, source_data["record_url"])
            self.assertEqual(
                {"raw.csv": hashlib.sha256(copied.read_bytes()).hexdigest()},
                source_data["files"],
            )
            # The fake retriever is not the hub: no manifest, so no file origin.
            self.assertIsNone(source_data["sources_manifest"])
            self.assertEqual(["raw.csv"], source_data["unrecorded"])
            package = source_data["dataset_package"]
            self.assertEqual("data/processor.py", package["processor"]["path"])
            self.assertEqual(
                (request.output_dir / "processor.py").read_bytes(),
                (request.output_dir / "data" / "processor.py").read_bytes(),
            )
            draft = json.loads(
                (request.output_dir / "data" / "dcat-ap-metadata.jsonld").read_text(
                    encoding="utf-8"
                )
            )
            dataset_node, processor_node = draft["@graph"]
            self.assertEqual("dcat:Dataset", dataset_node["@type"])
            self.assertEqual("#dataset", dataset_node["@id"])
            self.assertEqual(
                {"@id": request.dataset.spec.url}, dataset_node["prov:wasDerivedFrom"]
            )
            self.assertTrue(dataset_node["dct:title"].endswith(" (AI-ready)"))
            self.assertEqual("#processor", processor_node["@id"])
            self.assertEqual(
                [{"@id": processor_node["@id"]}], dataset_node["dct:relation"]
            )
            self.assertEqual("rdfs:Resource", processor_node["@type"])
            self.assertEqual("processor.py", processor_node["dcat:downloadURL"])
            self.assertEqual(
                {"@id": "https://data.hammerhai.eu/vocab/Processor"},
                processor_node["dct:conformsTo"],
            )
            self.assertEqual(
                {"checker": "fake-processor-check", "reproduced": ["test.csv", "train.csv"], "runtime": "fake"},
                package["processor_check"],
            )

    def test_pipeline_runs_url_derived_spec(self):
        retriever = FakeRetriever()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prompt = root / "agent.md"
            prompt.write_text("System prompt", encoding="utf-8")
            pipeline = DataEngineeringPipeline(
                retriever=retriever,
                harness=FakeHarness(),
                metadata_generator=FakeMetadataGenerator(),
                processor_checker=FakeProcessorChecker(),
                config=PipelineConfig(
                    workspace_root=root,
                    prompt_path=prompt,
                    model=ModelConfig("provider", "model"),
                ),
            )
            spec = dataset_spec_from_url(
                "https://catalog.example.org/dataset/example"
            )

            result = pipeline.run_spec(spec)

            self.assertEqual(spec, result.dataset.spec)
            self.assertEqual(spec, retriever.calls[0][0])
            self.assertEqual(root / "data" / spec.key, result.dataset.data_dir)

    def test_repeated_runs_use_distinct_provider_model_output_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prompt = root / "agent.md"
            prompt.write_text("System prompt", encoding="utf-8")
            pipeline = DataEngineeringPipeline(
                retriever=FakeRetriever(),
                harness=FakeHarness(),
                metadata_generator=FakeMetadataGenerator(),
                processor_checker=FakeProcessorChecker(),
                config=PipelineConfig(
                    workspace_root=root,
                    prompt_path=prompt,
                    model=ModelConfig("gwdg", "devstral/model"),
                ),
            )

            first = pipeline.run_spec(CHEMICAL_SPEC)
            second = pipeline.run_spec(CHEMICAL_SPEC)

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
                processor_checker=FakeProcessorChecker(),
                config=PipelineConfig(
                    workspace_root=root,
                    prompt_path=prompt,
                    model=ModelConfig("provider", "model"),
                ),
            )

            results = pipeline.run_examples()

            self.assertEqual(len(EXAMPLE_DATASETS), len(results))
            self.assertEqual(set(EXAMPLE_DATASETS), {result.dataset.spec.key for result in results})
            self.assertEqual(len(EXAMPLE_DATASETS), len(harness.calls))
            self.assertEqual(len(EXAMPLE_DATASETS), len(metadata_generator.calls))

    def test_pipeline_rejects_processor_without_process_entry_point(self):
        class NoProcessHarness(FakeHarness):
            def run(self, request, model):
                result = super().run(request, model)
                (request.output_dir / "processor.py").write_text(
                    "def parse(data_dir, target_col):\n    pass\n", encoding="utf-8"
                )
                return result

        metadata_generator = FakeMetadataGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prompt = root / "agent.md"
            prompt.write_text("System prompt", encoding="utf-8")
            pipeline = DataEngineeringPipeline(
                retriever=FakeRetriever(),
                harness=NoProcessHarness(),
                metadata_generator=metadata_generator,
                processor_checker=FakeProcessorChecker(),
                config=PipelineConfig(
                    workspace_root=root,
                    prompt_path=prompt,
                    model=ModelConfig("provider", "model"),
                ),
            )

            with self.assertRaisesRegex(ValueError, "process\\(data_dir, output_dir\\)"):
                pipeline.run_spec(TCM_SPEC)
            self.assertEqual([], metadata_generator.calls)

    def test_pipeline_rejects_false_harness_completion_before_metadata(self):
        class MissingOutputHarness:
            @property
            def name(self):
                return "missing-output"

            def run(self, request, _model):
                (request.output_dir / "processor.py").write_text(
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
                processor_checker=FakeProcessorChecker(),
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
                pipeline.run_spec(TCM_SPEC)
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
                processor_checker=FakeProcessorChecker(),
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
                pipeline.run_spec(CHEMICAL_SPEC)

            self.assertEqual([], harness.calls)


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
                for name in ("processor.py", "train.csv", "test.csv"):
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
            spec = INDUSTRY5_SPEC
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
            self.assertTrue((output_dir / "agent_transcript.md").is_file())

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
                    for name in ("processor.py", "train.csv", "test.csv"):
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
            spec = TCM_SPEC
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
            spec = TCM_SPEC
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
                (output_dir / "agent_transcript.md").is_file()
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
                for name in ("processor.py", "train.csv", "test.csv"):
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
            spec = TCM_SPEC
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
                for name in ("processor.py", "train.csv", "test.csv"):
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
            spec = CHEMICAL_SPEC
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
            spec = CHEMICAL_SPEC
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
