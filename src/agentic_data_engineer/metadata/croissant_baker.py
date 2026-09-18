import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from croissant_baker.metadata_generator import MetadataGenerator

from ..contracts import (
    MetadataGenerationResult,
    RetrievedDataset,
)
from ..validation import test_croissant_all

BakerFactory = Callable[..., Any]
CroissantValidator = Callable[[str | Path], None]

_LICENSE_ALIASES = {
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
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
    if parsed.netloc.lower() != "zenodo.org" or "/author/" not in parsed.path:
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
    citation = (
        f"https://doi.org/{identifier}"
        if identifier and identifier.startswith("10.")
        else identifier
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
        "date_published": _literal(dataset_node.get("dct:issued")),
        "date_modified": _literal(dataset_node.get("dct:modified")),
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

        generator = self._generator_factory(
            dataset_path=str(output_dir),
            includes=["train.csv", "test.csv"],
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
