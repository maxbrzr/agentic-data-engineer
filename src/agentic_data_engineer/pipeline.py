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
    ProcessorChecker,
)
from .retrieval import get_example, list_examples

from dcat_ap_hub import add_processor, ai_ready_record
from dcat_ap_hub.metadata.constants import (
    METADATA_FILENAME,
    PROCESSOR_FILENAME,
    SOURCES_MANIFEST,
    SPLIT_FILES,
)
from dcat_ap_hub.metadata.transfer import read_sources_manifest

SOURCE_DATA_DIR_NAME = "data"
DATASET_METADATA_NAME = METADATA_FILENAME
# Artifacts every successful agent run must leave in its output directory.
REQUIRED_AGENT_OUTPUTS = (PROCESSOR_FILENAME, *SPLIT_FILES)


class DataEngineeringPipeline:
    """Harness- and model-neutral orchestration for retrieval and processing."""

    def __init__(
        self,
        *,
        retriever: DatasetRetriever,
        harness: AgentHarness,
        metadata_generator: DatasetMetadataGenerator,
        processor_checker: ProcessorChecker,
        config: PipelineConfig,
    ) -> None:
        self.retriever = retriever
        self.harness = harness
        self.metadata_generator = metadata_generator
        self.processor_checker = processor_checker
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
        source_data = self.copy_source_data(
            retrieved.data_dir,
            output_dir,
            record_url=retrieved.spec.url,
            workspace_root=self.config.workspace_root,
        )
        package = self.build_dataset_package(output_dir, retrieved.spec)
        # Fails the run unless dcat-ap-hub would reproduce the exported split.
        check = self.processor_checker.check(
            output_dir / SOURCE_DATA_DIR_NAME,
            {name: output_dir / name for name in SPLIT_FILES},
        )
        package["processor_check"] = {
            "checker": check.checker,
            "reproduced": sorted(check.sha256),
            "runtime": check.runtime,
        }
        source_data["dataset_package"] = package
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
            f"Analyze the dataset, create the reusable processor at {output_dir / PROCESSOR_FILENAME}, "
            "run all applicable tabular validations, and export validated train/test "
            "splits. The pipeline generates and validates croissant.json after your run; "
            "do not create or edit that file. Do not report success unless processor.py, "
            "train.csv, and test.csv exist, process() in processor.py reproduces them, "
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
            for name in REQUIRED_AGENT_OUTPUTS
            if not (output_dir / name).is_file()
        ]
        if missing:
            raise ValueError(
                f"Agent run {run_id!r} reported completion, but required host "
                f"artifacts are missing from {output_dir}: {missing}."
            )
        if not DataEngineeringPipeline._defines_process(output_dir / PROCESSOR_FILENAME):
            raise ValueError(
                f"Agent run {run_id!r} reported completion, but processor.py does not "
                "define the module-level process(data_dir, output_dir) function "
                "required by dcat-ap-hub."
            )

    @staticmethod
    def _defines_process(processor_path: Path) -> bool:
        """Check statically (without executing agent code) for process()."""
        try:
            tree = ast.parse(processor_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            return False
        return any(
            isinstance(node, ast.FunctionDef)
            and node.name == "process"
            and len(node.args.args) == 2
            for node in tree.body
        )

    @classmethod
    def copy_source_data(
        cls,
        data_dir: Path,
        output_dir: Path,
        *,
        record_url: str | None = None,
        workspace_root: Path | None = None,
    ) -> dict:
        """
        Copy the original dataset into <output_dir>/data and record its origin.

        Besides a hash per file, this records the catalog record and, from the
        hub's sources manifest, the URL every file was downloaded from. Files
        the manifest does not account for (placed by hand, or a folder not
        downloaded by the hub) are listed as ``unrecorded``; downloads that
        failed or returned an HTML page as ``failed_sources``.
        """
        target = output_dir / SOURCE_DATA_DIR_NAME
        shutil.copytree(data_dir, target, dirs_exist_ok=True)
        files = {
            str(file.relative_to(target)): cls._sha256(file)
            for file in sorted(target.rglob("*"))
            if file.is_file()
        }
        manifest = read_sources_manifest(target)
        sidecars = {DATASET_METADATA_NAME, SOURCES_MANIFEST}
        origins: dict[str, str] = {}
        failed: list[dict] = []
        if manifest is not None:
            sources = manifest.get("sources", [])
            for name, entry in manifest.get("files", {}).items():
                source = sources[entry["source"]]
                if name in files and entry.get("sha256") == files[name]:
                    origins[name] = source["url"]
            failed = [
                {key: source.get(key) for key in ("title", "url", "status", "error") if source.get(key)}
                for source in sources
                if source.get("status") != "downloaded"
            ]
        try:
            copied_from = str(data_dir.resolve().relative_to(workspace_root.resolve())) if workspace_root else str(data_dir)
        except ValueError:
            copied_from = str(data_dir)
        return {
            "path": SOURCE_DATA_DIR_NAME,
            "record_url": record_url,
            "copied_from": copied_from,
            "sources_manifest": SOURCES_MANIFEST if manifest is not None else None,
            "files": files,
            "origins": origins,
            "unrecorded": sorted(name for name in files if name not in origins and name not in sidecars),
            "failed_sources": failed,
        }

    @classmethod
    def build_dataset_package(cls, output_dir: Path, spec: DatasetSpec) -> dict:
        """
        Make <output_dir>/data a dcat-ap-hub dataset folder.

        Adds a copy of processor.py and a draft dcat-ap-metadata.jsonld for the
        AI-ready dataset, built with dcat-ap-hub: ``ai_ready_record`` gives it
        its own id and ``prov:wasDerivedFrom`` the raw dataset, and
        ``add_processor`` declares the processor. So
        ``Dataset.from_directory(<run>/data).process()`` works before anything
        is published. The id and processor URL are relative: they stay valid
        when the folder is published as is.
        """
        data_dir = output_dir / SOURCE_DATA_DIR_NAME
        processor = data_dir / PROCESSOR_FILENAME
        shutil.copy2(output_dir / PROCESSOR_FILENAME, processor)
        metadata_path = data_dir / DATASET_METADATA_NAME
        try:
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            document = {}
        try:
            document = ai_ready_record(document, raw_id=spec.url)
        except ValueError:
            # The raw record had no dataset node; derive from the catalog URL.
            raw = {"@graph": [{"@id": spec.url, "@type": "dcat:Dataset", "dct:title": spec.title}]}
            document = ai_ready_record(raw)
        document = add_processor(document)
        metadata_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
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
        for name in REQUIRED_AGENT_OUTPUTS:
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
