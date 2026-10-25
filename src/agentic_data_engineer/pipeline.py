import ast
import hashlib
import json
import re
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .config import PipelineConfig
from .contracts import (
    AgentHarness,
    AgentRequest,
    DatasetMetadataGenerator,
    DatasetRetriever,
    DatasetSpec,
    PipelineRunResult,
)
from .retrieval import get_example, list_examples

SOURCE_DATA_DIR_NAME = "data"
DATASET_METADATA_NAME = "dcat-metadata.jsonld"
# dcat-ap-hub's processor profile; marks parser.py as the dataset's processor.
PROCESSOR_PROFILE_URI = "http://example.org/profiles/Processor"


class DataEngineeringPipeline:
    """Harness- and model-neutral orchestration for retrieval and processing."""

    def __init__(
        self,
        *,
        retriever: DatasetRetriever,
        harness: AgentHarness,
        metadata_generator: DatasetMetadataGenerator,
        config: PipelineConfig,
    ) -> None:
        self.retriever = retriever
        self.harness = harness
        self.metadata_generator = metadata_generator
        self.config = config

    def run(self, example_key: str, *, force_download: bool | None = None) -> PipelineRunResult:
        return self.run_spec(
            get_example(example_key),
            force_download=force_download,
        )

    def run_spec(
        self,
        spec: DatasetSpec,
        *,
        force_download: bool | None = None,
    ) -> PipelineRunResult:
        """Run the pipeline for a configured example or URL-derived spec."""
        retrieved = self.retriever.retrieve(
            spec,
            self.config.data_root,
            force=self.config.force_download if force_download is None else force_download,
        )
        self.metadata_generator.preflight(retrieved)

        output_dir = self._create_run_output_dir(spec.key)
        provenance_path = output_dir / "provenance.json"
        provenance_path.unlink(missing_ok=True)
        request = AgentRequest(
            dataset=retrieved,
            output_dir=output_dir,
            system_prompt=self.config.load_system_prompt(),
            task_prompt=self._build_task_prompt(retrieved.data_dir, output_dir),
        )
        agent_result = self.harness.run(request, self.config.model)
        self._require_agent_outputs(output_dir, agent_result.run_id)
        metadata_result = self.metadata_generator.generate(retrieved, output_dir)
        # Copied only after the agent and metadata steps, so the agent never
        # gets a writable copy of the raw data.
        source_data = self.copy_source_data(retrieved.data_dir, output_dir)
        source_data["dataset_package"] = self.build_dataset_package(
            output_dir, retrieved.spec
        )
        self._write_provenance(
            provenance_path,
            retrieved,
            agent_result,
            metadata_result,
            source_data,
        )
        return PipelineRunResult(
            dataset=retrieved,
            agent=agent_result,
            metadata=metadata_result,
            provenance_path=provenance_path,
        )

    def run_examples(
        self,
        example_keys: Iterable[str] | None = None,
        *,
        force_download: bool | None = None,
    ) -> tuple[PipelineRunResult, ...]:
        keys = (
            tuple(example_keys)
            if example_keys is not None
            else tuple(spec.key for spec in list_examples())
        )
        return tuple(
            self.run(key, force_download=force_download)
            for key in keys
        )

    @staticmethod
    def _build_task_prompt(data_dir, output_dir) -> str:
        return (
            "Follow the system instructions completely. "
            f"The read-only local dataset is in {data_dir}. "
            f"Write every generated artifact to {output_dir}. "
            f"Analyze the dataset, create the reusable parser at {output_dir / 'parser.py'}, "
            "run all applicable tabular validations, and export validated train/test "
            "splits. The pipeline generates and validates croissant.json after your run; "
            "do not create or edit that file. Do not report success unless parser.py, "
            "train.csv, and test.csv exist, process() in parser.py reproduces them, "
            "and every applicable tabular validation passes."
        )

    def _create_run_output_dir(self, example_key: str) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        provider = self._path_component(self.config.model.provider_id)
        model = self._path_component(self.config.model.model_id)
        run_name = f"{timestamp}__{provider}__{model}__{uuid4().hex[:8]}"
        output_dir = self.config.output_root / example_key / "runs" / run_name
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir

    @staticmethod
    def _path_component(value: str) -> str:
        component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
        return (component or "unnamed")[:80]

    @staticmethod
    def _require_agent_outputs(output_dir, run_id: str) -> None:
        """Reject harness completion unless required artifacts reached the host."""
        missing = [
            name
            for name in ("parser.py", "train.csv", "test.csv")
            if not (output_dir / name).is_file()
        ]
        if missing:
            raise ValueError(
                f"Agent run {run_id!r} reported completion, but required host "
                f"artifacts are missing from {output_dir}: {missing}."
            )
        if not DataEngineeringPipeline._defines_process(output_dir / "parser.py"):
            raise ValueError(
                f"Agent run {run_id!r} reported completion, but parser.py does not "
                "define the module-level process(input_files, output_dir) function "
                "required by dcat-ap-hub."
            )

    @staticmethod
    def _defines_process(parser_path: Path) -> bool:
        """Check statically (without executing agent code) for process()."""
        try:
            tree = ast.parse(parser_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            return False
        return any(
            isinstance(node, ast.FunctionDef)
            and node.name == "process"
            and len(node.args.args) >= 2
            for node in tree.body
        )

    @classmethod
    def copy_source_data(cls, data_dir: Path, output_dir: Path) -> dict:
        """Copy the original dataset into <output_dir>/data and hash each file."""
        target = output_dir / SOURCE_DATA_DIR_NAME
        shutil.copytree(data_dir, target, dirs_exist_ok=True)
        return {
            "path": SOURCE_DATA_DIR_NAME,
            "copied_from": str(data_dir),
            "files": {
                str(file.relative_to(target)): cls._sha256(file)
                for file in sorted(target.rglob("*"))
                if file.is_file()
            },
        }

    @classmethod
    def build_dataset_package(cls, output_dir: Path, spec: DatasetSpec) -> dict:
        """
        Make <output_dir>/data a dcat-ap-hub dataset folder.

        Adds a copy of parser.py and a draft dcat-metadata.jsonld that declares it
        as the dataset's processor, so ``Dataset.from_directory(<run>/data)``
        followed by ``.process()`` works before the data is published. The
        published metadata replaces the draft.
        """
        data_dir = output_dir / SOURCE_DATA_DIR_NAME
        processor = data_dir / "parser.py"
        shutil.copy2(output_dir / "parser.py", processor)
        metadata_path = data_dir / DATASET_METADATA_NAME
        try:
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            document = {}
        metadata_path.write_text(
            json.dumps(cls._with_processor(document, spec), indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "processor": {
                "path": f"{SOURCE_DATA_DIR_NAME}/{processor.name}",
                "sha256": cls._sha256(processor),
            },
            "metadata": {
                "path": f"{SOURCE_DATA_DIR_NAME}/{metadata_path.name}",
                "status": "draft; replaced by the published metadata",
                "sha256": cls._sha256(metadata_path),
            },
        }

    @staticmethod
    def _with_processor(document: dict, spec: DatasetSpec) -> dict:
        graph = document.get("@graph")
        if not isinstance(graph, list):
            graph = [document] if document.get("@type") else []
        dataset = next(
            (
                node
                for node in graph
                if isinstance(node, dict)
                and "dcat:Dataset" in (
                    node.get("@type")
                    if isinstance(node.get("@type"), list)
                    else [node.get("@type")]
                )
            ),
            None,
        )
        if dataset is None:
            dataset = {"@id": spec.url, "@type": "dcat:Dataset", "dct:title": spec.title}
            graph.insert(0, dataset)
        processor_id = f"{dataset.get('@id') or spec.url}#processor"
        graph = [
            node
            for node in graph
            if not (isinstance(node, dict) and node.get("@id") == processor_id)
        ]
        graph.append(
            {
                "@id": processor_id,
                "@type": "rdfs:Resource",
                "dct:title": "parser.py",
                "dct:description": "Generated processor producing train.csv and test.csv.",
                "dct:format": "text/x-python",
                "dct:conformsTo": PROCESSOR_PROFILE_URI,
                "dcat:downloadURL": "parser.py",
            }
        )
        relations = dataset.get("dct:relation", [])
        relations = relations if isinstance(relations, list) else [relations]
        if not any(
            isinstance(item, dict) and item.get("@id") == processor_id for item in relations
        ):
            relations.append({"@id": processor_id})
        dataset["dct:relation"] = relations
        return {**document, "@graph": graph}

    def _write_provenance(
        self,
        path: Path,
        dataset,
        agent_result,
        metadata_result,
        source_data: dict,
    ) -> None:
        model = self.config.model
        agent_identity = {
            "harness": agent_result.harness,
            "provider": model.provider_id,
            "model": model.model_id,
            "run_id": agent_result.run_id,
            "message_id": agent_result.message_id,
        }
        artifacts: dict[str, dict] = {}
        for name in ("parser.py", "train.csv", "test.csv"):
            artifact_path = path.parent / name
            artifacts[name] = {
                "producer_type": "agent",
                **agent_identity,
                "sha256": self._sha256(artifact_path),
            }

        for role, metadata_key in (
            ("run_log", "run_log"),
            ("run_report", "report"),
        ):
            value = agent_result.metadata.get(metadata_key)
            if not isinstance(value, str):
                continue
            artifact_path = Path(value).expanduser().resolve()
            if not artifact_path.is_file() or artifact_path.parent != path.parent:
                continue
            artifacts[artifact_path.name] = {
                "producer_type": "harness",
                "artifact_role": role,
                **agent_identity,
                "sha256": self._sha256(artifact_path),
            }

        metadata_path = metadata_result.path.expanduser().resolve()
        artifacts[metadata_path.name] = {
            "producer_type": "metadata_generator",
            "generator": metadata_result.generator,
            "derived_from_agent_run": agent_identity,
            "sha256": self._sha256(metadata_path),
        }

        document = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": {
                "key": dataset.spec.key,
                "title": dataset.spec.title,
                "catalog_url": dataset.spec.url,
            },
            "agent_run": agent_identity,
            "artifacts": artifacts,
            "source_data": source_data,
        }
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
