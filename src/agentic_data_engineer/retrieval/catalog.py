from types import MappingProxyType

from ..contracts import DatasetSpec

_EXAMPLE_DATASETS = {
    "tcm-predictive-maintenance": DatasetSpec(
        key="tcm-predictive-maintenance",
        title="TCM: Benchmark Datasets for Predictive Maintenance in Steel Manufacturing",
        url="http://data.sdil.de/en/dataset/10-5281-zenodo-11469702",
    ),
    "chemical-process-safety": DatasetSpec(
        key="chemical-process-safety",
        title="AI-Ready Simulation Dataset for Chemical Process Safety and Accident Prevention",
        url="http://data.sdil.de/en/dataset/10-5281-zenodo-18200704",
    ),
    "industry-5-cyber-physical-systems": DatasetSpec(
        key="industry-5-cyber-physical-systems",
        title=(
            "Smart Cyber-Physical Systems Dataset for Real-Time Optimization, "
            "Resilience, and Sustainability in Industry 5.0"
        ),
        url="http://data.sdil.de/en/dataset/10-5281-zenodo-18177239",
    ),
    "floor-type-detection": DatasetSpec(
        key="floor-type-detection",
        title="Floor Type Detection Dataset",
        url=(
            "http://blue.data.sdil.de/de/dataset/"
            "https-darus-uni-stuttgart-de-dataset-xhtml-"
            "persistentid-doi-10-18419-darus-4353"
        ),
    ),
    "printed-paper-scratches": DatasetSpec(
        key="printed-paper-scratches",
        title=(
            "Machine Learning based scratches on printed paper detection, "
            "in high-speed printing systems [Dataset]"
        ),
        url="https://data.sdil.de/en/dataset/10-5281-zenodo-6021701",
    ),
    "mimii-sound-anomaly-detection": DatasetSpec(
        key="mimii-sound-anomaly-detection",
        title=(
            "MIMII Dataset: Sound Dataset for Malfunctioning Industrial "
            "Machine Investigation and Inspection"
        ),
        url="http://data.sdil.de/en/dataset/10-5281-zenodo-3384388",
    ),
    "mimii-dg": DatasetSpec(
        key="mimii-dg",
        title=(
            "MIMII DG: Sound Dataset for Malfunctioning Industrial Machine "
            "Investigation for Domain Generalization Task"
        ),
        url="https://zenodo.org/records/6355122",
        metadata_resource="metadata/mimii-dg.jsonld",
    ),
    "bearing": DatasetSpec(
        key="bearing",
        title="DCASE 2022 Task 2 Development Dataset: Bearing subset",
        url="https://zenodo.org/records/6355122",
        metadata_resource="metadata/mimii-dg-bearing.jsonld",
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
