from collections.abc import Iterable

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

        output_dir = self.config.output_root / spec.key
        output_dir.mkdir(parents=True, exist_ok=True)
        request = AgentRequest(
            dataset=retrieved,
            output_dir=output_dir,
            system_prompt=self.config.load_system_prompt(),
            task_prompt=self._build_task_prompt(retrieved.data_dir, output_dir),
        )
        agent_result = self.harness.run(request, self.config.model)
        self._require_agent_outputs(output_dir, agent_result.run_id)
        metadata_result = self.metadata_generator.generate(retrieved, output_dir)
        return PipelineRunResult(
            dataset=retrieved,
            agent=agent_result,
            metadata=metadata_result,
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
