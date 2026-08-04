import hashlib
import json
import re
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
    PipelineRunResult,
)
from .retrieval import get_example, list_examples


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
        spec = get_example(example_key)
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
        self._write_provenance(
            provenance_path,
            retrieved,
            agent_result,
            metadata_result,
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
            "train.csv, and test.csv exist and every applicable tabular validation passes."
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

    def _write_provenance(
        self,
        path: Path,
        dataset,
        agent_result,
        metadata_result,
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
