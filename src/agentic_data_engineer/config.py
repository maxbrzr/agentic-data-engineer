from dataclasses import dataclass
from pathlib import Path

from .contracts import ModelConfig


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Filesystem and model configuration for the current local pipeline."""

    workspace_root: Path
    prompt_path: Path
    model: ModelConfig
    force_download: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_root", self.workspace_root.expanduser().resolve())
        object.__setattr__(self, "prompt_path", self.prompt_path.expanduser().resolve())

    @property
    def data_root(self) -> Path:
        return self.workspace_root / "data"

    @property
    def output_root(self) -> Path:
        return self.workspace_root / "output"

    def load_system_prompt(self) -> str:
        if not self.prompt_path.is_file():
            raise FileNotFoundError(f"Agent prompt does not exist: {self.prompt_path}")
        return self.prompt_path.read_text(encoding="utf-8")
