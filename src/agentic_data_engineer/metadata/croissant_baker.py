import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from croissant_baker.handlers.registry import register_handler
from croissant_baker.metadata_generator import MetadataGenerator

from ..contracts import (
    MetadataGenerationResult,
    RetrievedDataset,
)
from ..validation import test_croissant_all
from .audio_handler import AudioHandler

_AUDIO_HANDLER = AudioHandler()
register_handler(_AUDIO_HANDLER)

BakerFactory = Callable[..., Any]
CroissantValidator = Callable[[str | Path], None]

_LICENSE_ALIASES = {
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "cc-by-nd-4.0": "CC-BY-ND-4.0",
    "cc0-1.0": "CC0-1.0",
}
_REQUIRED_SOURCE_FIELDS = (
    "name",
    "description",
    "license",
    "citation",
    "date_published",
    "creators",
)
_IMAGE_SUFFIXES = (
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)
_MEDIA_SPLITS = ("train", "test", "validation")
_IMAGE_DIRECTORIES = ("images", "masks")
_IMAGE_SIDECARS = (
    "labels.csv",
    "mask_labels.csv",
    "train_labels.csv",
    "test_labels.csv",
    "validation_labels.csv",
    "train_objects.csv",
    "test_objects.csv",
    "validation_objects.csv",
    "train_captions.csv",
    "test_captions.csv",
    "validation_captions.csv",
)
_AUDIO_SUFFIXES = (
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
)
_AUDIO_SIDECARS = (
    "labels.csv",
    "train_labels.csv",
    "test_labels.csv",
    "validation_labels.csv",
    "train_segments.csv",
    "test_segments.csv",
    "validation_segments.csv",
    "train_transcripts.csv",
    "test_transcripts.csv",
    "validation_transcripts.csv",
)


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _literal(value: Any) -> str | None:
    candidates = _values(value)
    localized = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("@language") == "en"
    ]
    for candidate in localized + candidates:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, dict):
            literal = candidate.get("@value") or candidate.get("@id")
            if literal is not None:
                return str(literal)
    return None


def _node_has_type(node: dict[str, Any], expected: str) -> bool:
    node_types = {
        str(node_type)
        for node_type in _values(node.get("@type"))
    }
    return expected in node_types or any(
        node_type.endswith(f"/{expected.split(':')[-1]}")
        or node_type.endswith(f"#{expected.split(':')[-1]}")
        for node_type in node_types
    )


def _creator_name(
    creator: Any,
    nodes_by_id: dict[str, dict[str, Any]],
) -> str | None:
    if isinstance(creator, str):
        return creator
    if not isinstance(creator, dict):
        return None

    identifier = creator.get("@id")
    resolved = nodes_by_id.get(str(identifier)) if identifier else None
    node = resolved or creator
    for field in ("foaf:name", "schema:name", "name", "dct:title"):
        if name := _literal(node.get(field)):
            return name

    if not identifier:
        return None
    parsed = urlparse(str(identifier))
    if "/author/" not in parsed.path:
        return None

    encoded_name = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    name = unquote(encoded_name).replace("_", " ").strip()
    name = re.sub(r"\s*,\s*", ", ", name)
    name = re.sub(r"\s+", " ", name)
    return name or None


def _load_dcat_metadata(dataset: RetrievedDataset) -> dict[str, Any]:
    metadata_paths = sorted(dataset.data_dir.rglob("dcat-metadata.jsonld"))
    if not metadata_paths:
        raise ValueError(
            f"No dcat-metadata.jsonld found under {dataset.data_dir}; "
            "Croissant metadata must not be generated from invented defaults."
        )
    if len(metadata_paths) > 1:
        paths = ", ".join(str(path) for path in metadata_paths)
        raise ValueError(f"Multiple DCAT metadata files found: {paths}")

    with metadata_paths[0].open(encoding="utf-8") as stream:
        document = json.load(stream)
    graph = document.get("@graph")
    if not isinstance(graph, list):
        raise ValueError(f"DCAT metadata has no @graph: {metadata_paths[0]}")

    nodes = [node for node in graph if isinstance(node, dict)]
    dataset_nodes = [
        node for node in nodes if _node_has_type(node, "dcat:Dataset")
    ]
    if len(dataset_nodes) != 1:
        raise ValueError(
            "Expected exactly one dcat:Dataset node, "
            f"found {len(dataset_nodes)} in {metadata_paths[0]}."
        )

    dataset_node = dataset_nodes[0]
    nodes_by_id = {
        str(node["@id"]): node
        for node in nodes
        if node.get("@id") is not None
    }

    creator_names = []
    for creator in _values(dataset_node.get("dct:creator")):
        name = _creator_name(creator, nodes_by_id)
        if name:
            creator_names.append(name)

    licenses = {
        license_value
        for node in [dataset_node, *nodes]
        if (license_value := _literal(node.get("dct:license")))
    }
    if len(licenses) > 1:
        raise ValueError(f"Conflicting DCAT licenses found: {sorted(licenses)}")

    keywords = [
        keyword
        for value in _values(dataset_node.get("dcat:keyword"))
        if (keyword := _literal(value))
    ]
    identifier = _literal(dataset_node.get("dct:identifier"))
    landing_page = _literal(dataset_node.get("dcat:landingPage"))
    if not identifier and landing_page:
        match = re.search(r"(?:persistentId=)?doi:(10\.[^&]+)", landing_page)
        identifier = match.group(1) if match else None
    citation = (
        f"https://doi.org/{identifier}"
        if identifier and identifier.startswith("10.")
        else identifier or landing_page
    )

    return {
        "name": _literal(dataset_node.get("dct:title")) or dataset.metadata_title,
        "description": _literal(dataset_node.get("dct:description")),
        "url": dataset.spec.url,
        "license": (
            _LICENSE_ALIASES.get(next(iter(licenses)).lower(), next(iter(licenses)))
            if licenses
            else None
        ),
        "citation": citation,
        "date_published": _literal(
            dataset_node.get("dct:issued") or dataset_node.get("dcat:issued")
        ),
        "date_modified": _literal(
            dataset_node.get("dct:modified") or dataset_node.get("dcat:modified")
        ),
        "creators": [{"name": name} for name in creator_names],
        "keywords": keywords or None,
    }


def _require_source_metadata(metadata: dict[str, Any]) -> None:
    missing = [
        field for field in _REQUIRED_SOURCE_FIELDS if not metadata.get(field)
    ]
    if missing:
        raise ValueError(
            "DCAT metadata is missing values required for Croissant generation: "
            f"{missing}."
        )


def _contains_supported_image(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        for path in directory.rglob("*")
    )


def _image_baker_includes(output_dir: Path) -> list[str]:
    """Return image-related includes for a standardized image run, if present."""
    present_splits = {
        split
        for split in _MEDIA_SPLITS
        if _contains_supported_image(output_dir / split / "images")
    }
    if not present_splits:
        return []

    missing = {"train", "test"} - present_splits
    if missing:
        raise ValueError(
            "Cannot generate image Croissant metadata; image directories are "
            f"missing supported files for splits: {sorted(missing)}."
        )

    includes: list[str] = []
    for name in _IMAGE_SIDECARS:
        if (output_dir / name).is_file():
            includes.append(name)

    for split in _MEDIA_SPLITS:
        if split not in present_splits:
            continue
        validation_manifest = output_dir / f"{split}.csv"
        if split == "validation" and validation_manifest.is_file():
            includes.append(validation_manifest.name)
        for directory in _IMAGE_DIRECTORIES:
            root = output_dir / split / directory
            if not root.is_dir():
                continue
            for suffix in _IMAGE_SUFFIXES:
                includes.extend(
                    (
                        f"{split}/{directory}/*{suffix}",
                        f"{split}/{directory}/**/*{suffix}",
                    )
                )
    return includes


def _audio_baker_includes(output_dir: Path) -> list[str]:
    """Return audio-related includes for a standardized audio run, if present."""
    files_by_split = {
        split: _supported_audio_files(output_dir / split / "audio")
        for split in _MEDIA_SPLITS
    }
    present_splits = {
        split for split, paths in files_by_split.items() if paths
    }
    if not present_splits:
        return []

    missing = {"train", "test"} - present_splits
    if missing:
        raise ValueError(
            "Cannot generate audio Croissant metadata; audio directories are "
            f"missing supported files for splits: {sorted(missing)}."
        )

    includes = [
        name for name in _AUDIO_SIDECARS if (output_dir / name).is_file()
    ]
    for split in _MEDIA_SPLITS:
        if split not in present_splits:
            continue
        validation_manifest = output_dir / f"{split}.csv"
        if split == "validation" and validation_manifest.is_file():
            includes.append(validation_manifest.name)
        invalid = [
            path for path in files_by_split[split] if not _AUDIO_HANDLER.can_handle(path)
        ]
        if invalid:
            relative = [path.relative_to(output_dir).as_posix() for path in invalid]
            raise ValueError(
                "Cannot generate audio Croissant metadata; files have an "
                f"unsupported or invalid audio signature: {relative}."
            )
        includes.extend(
            path.relative_to(output_dir).as_posix()
            for path in files_by_split[split]
        )
    return includes


def _supported_audio_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES
    )


class CroissantBakerMetadataGenerator:
    """Generate deterministic Croissant metadata after agent processing."""

    def __init__(
        self,
        *,
        artifact_version: str = "1.0.0",
        generator_factory: BakerFactory = MetadataGenerator,
        validator: CroissantValidator = test_croissant_all,
    ) -> None:
        self.artifact_version = artifact_version
        self._generator_factory = generator_factory
        self._validator = validator

    @property
    def name(self) -> str:
        return "croissant-baker"

    def preflight(self, dataset: RetrievedDataset) -> None:
        """Fail before the agent run if source metadata cannot support Croissant."""
        _require_source_metadata(_load_dcat_metadata(dataset))

    def generate(
        self,
        dataset: RetrievedDataset,
        output_dir: Path,
    ) -> MetadataGenerationResult:
        output_dir = Path(output_dir).expanduser().resolve()
        required_files = [output_dir / "train.csv", output_dir / "test.csv"]
        missing = [path.name for path in required_files if not path.is_file()]
        if missing:
            raise ValueError(
                "Cannot generate Croissant metadata; missing agent outputs: "
                f"{missing}."
            )

        metadata = _load_dcat_metadata(dataset)
        _require_source_metadata(metadata)

        includes = ["train.csv", "test.csv"]
        includes.extend(_image_baker_includes(output_dir))
        includes.extend(_audio_baker_includes(output_dir))
        includes = list(dict.fromkeys(includes))

        generator = self._generator_factory(
            dataset_path=str(output_dir),
            includes=includes,
            name=metadata["name"],
            description=metadata["description"],
            url=metadata["url"],
            license=metadata["license"],
            citation=metadata["citation"],
            version=self.artifact_version,
            date_published=metadata["date_published"],
            date_modified=metadata["date_modified"],
            creators=metadata["creators"],
            keywords=metadata["keywords"],
        )

        final_path = output_dir / "croissant.json"
        temporary_path = output_dir / ".croissant.json.tmp"
        try:
            generator.save_metadata(str(temporary_path), validate=True)
            self._validator(temporary_path)
            temporary_path.replace(final_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        return MetadataGenerationResult(
            generator=self.name,
            path=final_path,
        )
