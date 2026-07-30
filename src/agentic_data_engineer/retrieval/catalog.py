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
}

EXAMPLE_DATASETS = MappingProxyType(_EXAMPLE_DATASETS)


def list_examples() -> tuple[DatasetSpec, ...]:
    """Return the only datasets currently enabled for pipeline runs."""
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
