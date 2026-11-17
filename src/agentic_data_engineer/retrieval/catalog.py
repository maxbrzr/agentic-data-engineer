from types import MappingProxyType

from ..contracts import DatasetSpec

_EXAMPLE_DATASETS = {
    "copper-rings-insertion-attempts": DatasetSpec(
        key="copper-rings-insertion-attempts",
        title="Copper Rings Insertion Attempts Dataset",
        url="https://data.hammerhai.eu/de/dataset/10-5281-zenodo-6364950",
        metadata_resource="metadata/copper-rings-insertion-attempts.jsonld",
    ),
    "critical-manufacturing": DatasetSpec(
        key="critical-manufacturing",
        title="Critical Manufacturing (local CSV)",
        url="urn:local:critical-manufacturing",
    ),
    "Vale - Semantic Terrain Segmentation": DatasetSpec(
        key="Vale - Semantic Terrain Segmentation",
        title="Vale - Semantic Terrain Segmentation",
        url="https://data.hammerhai.eu/de/dataset/452430",
        metadata_resource="metadata/image-segmentation.jsonld",
    ),
    "printed-paper-scratches": DatasetSpec(
        key="printed-paper-scratches",
        title=(
            "Machine Learning based scratches on printed paper detection, "
            "in high-speed printing systems [Dataset]"
        ),
        url="https://data.sdil.de/en/dataset/10-5281-zenodo-6021701",
    ),
    "bearing": DatasetSpec(
        key="bearing",
        title="DCASE 2022 Task 2 Development Dataset: Bearing subset",
        url="https://zenodo.org/records/6355122",
        metadata_resource="metadata/mimii-dg-bearing.jsonld",
    ),
    "machine-predictive-maintenance-classification": DatasetSpec(
        key="machine-predictive-maintenance-classification",
        title="Machine Predictive Maintenance Classification",
        url="https://data.hammerhai.eu/de/dataset/1697740",
        metadata_resource="metadata/machine-predictive-maintenance-classification.jsonld",
    ),
    "Workpieces image dataset": DatasetSpec(
        key="Workpieces image dataset",
        title="Workpieces image dataset",
        url="https://zenodo.org/records/16361102",
        metadata_resource="metadata/workpieces-image-dataset.jsonld",
    ),
}

EXAMPLE_DATASETS = MappingProxyType(_EXAMPLE_DATASETS)


def list_examples() -> tuple[DatasetSpec, ...]:
    """Return the datasets currently enabled for pipeline runs."""
    return tuple(EXAMPLE_DATASETS.values())


def get_example(key: str) -> DatasetSpec:
    """Resolve an enabled example by its stable key."""
    try:
        return EXAMPLE_DATASETS[key]
    except KeyError as exc:
        available = ", ".join(EXAMPLE_DATASETS)
        raise KeyError(
            f"Unknown example {key!r}. Available examples: {available}."
        ) from exc
