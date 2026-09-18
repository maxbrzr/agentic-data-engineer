from dataclasses import dataclass
from pathlib import Path

from .contracts import ModelConfig

_AUGMENTATION_PLACEHOLDER = "{{DATA_AUGMENTATION}}"
_AUGMENTATION_DISABLED = (
    "Data augmentation is disabled. Do not generate augmented observations or "
    "apply augmentation transformations. Follow the original parsing, "
    "preprocessing, splitting, copying and validation requirements unchanged."
)
_AUGMENTATION_ENABLED = """Data augmentation is enabled only as described by the subprompt below.
Determine original split membership before augmentation. Augment training data
only; never augment test or validation data and never modify raw source files.
Keep originals and all their derivatives within the same split. Track each
derived observation back to its source internally or in an audit sidecar.
Preserve the authoritative target and transform associated annotations only
when the subprompt explicitly defines a valid transformation for the task.
Source-preservation and byte-for-byte checks still apply to originals and all
evaluation data. For derived training observations, validate the documented
transformation, annotation consistency and lineage instead of requiring a hash
match with the original. Follow the remaining output and validation contracts.
Use a recorded seed and require repeatable outputs. Document techniques,
parameters, seed, lineage, validation results and original/augmented counts.

## Augmentation subprompt
"""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Filesystem and model configuration for the current local pipeline."""

    workspace_root: Path
    prompt_path: Path
    model: ModelConfig
    force_download: bool = False
    data_augmentation: bool = False

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
        prompt = self.prompt_path.read_text(encoding="utf-8")
        if prompt.count(_AUGMENTATION_PLACEHOLDER) > 1:
            raise ValueError("System prompt must contain at most one augmentation placeholder.")

        augmentation = _AUGMENTATION_DISABLED
        if self.data_augmentation:
            path = self.prompt_path.parent / "augmentation" / self.prompt_path.name
            if not path.is_file():
                raise FileNotFoundError(f"Augmentation subprompt does not exist: {path}")
            subprompt = path.read_text(encoding="utf-8").strip()
            if not subprompt:
                raise ValueError(f"Augmentation subprompt is empty: {path}")
            augmentation = _AUGMENTATION_ENABLED + subprompt

        if _AUGMENTATION_PLACEHOLDER in prompt:
            return prompt.replace(_AUGMENTATION_PLACEHOLDER, augmentation)
        # Custom prompts without the template marker also receive an explicit policy.
        return prompt.rstrip() + "\n\n# Data augmentation\n\n" + augmentation + "\n"

    def load_example_guidance(self, example_key: str) -> str:
        """Load optional task guidance without replacing the system prompt."""
        path = self.workspace_root / "prompts" / "examples" / f"{example_key}.md"
        return path.read_text(encoding="utf-8") if path.is_file() else ""
