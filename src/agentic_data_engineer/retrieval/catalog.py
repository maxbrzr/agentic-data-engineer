import hashlib
import re
from types import MappingProxyType
from urllib.parse import urlparse

from ..contracts import DatasetSpec

_EXAMPLE_DATASETS = {
    "edge-ai-industrial-robot-motion": DatasetSpec(
        key="edge-ai-industrial-robot-motion",
        title="Edge-AI Industrial Robot Motion Dataset",
        url="https://data.hammerhai.eu/en/dataset/7779224",
        local_dir_name="Edge-AI Industrial Robot Motion Dataset",
    ),
    "industrial-digital-twin-1-year": DatasetSpec(
        key="industrial-digital-twin-1-year",
        title="Industrial Digital Twin Dataset 1 Year",
        url="https://data.hammerhai.eu/en/dataset/9627351",
        local_dir_name="Industrial Digital Twin Dataset 1 Year",
    ),
    "industrial-machine-predictive-maintenance": DatasetSpec(
        key="industrial-machine-predictive-maintenance",
        title="Industrial Machine Predictive Maintenance (syn)",
        url=(
            "https://data.hammerhai.eu/en/dataset/"
            "tatheerabbas-industrial-machine-predictive-maintenance"
        ),
        local_dir_name="Industrial Machine Predictive Maintenance (syn)",
    ),
    "industrial-robotics-and-automation": DatasetSpec(
        key="industrial-robotics-and-automation",
        title="Industrial Robotics and Automation Dataset",
        url="https://data.hammerhai.eu/en/dataset/5973385",
        local_dir_name="Industrial Robotics and Automation Dataset",
    ),
    "industrial-smart-factory-automation-2026": DatasetSpec(
        key="industrial-smart-factory-automation-2026",
        title="Industrial Smart Factory Automation Dataset 2026",
        url=(
            "https://data.hammerhai.eu/en/dataset/"
            "rauffauzanrambe-industrial-smart-factory-automation-dataset-2026"
        ),
        local_dir_name="Industrial Smart Factory Automation Dataset 2026",
    ),
    "machine-failure": DatasetSpec(
        key="machine-failure",
        title="Machine Failure Dataset",
        url=(
            "https://data.hammerhai.eu/en/dataset/"
            "sumit913-machine-failure-dataset"
        ),
        local_dir_name="Machine Failure Dataset",
    ),
    "machine-predictive-maintenance-classification": DatasetSpec(
        key="machine-predictive-maintenance-classification",
        title="Machine Predictive Maintenance Classification",
        url="https://data.hammerhai.eu/en/dataset/1697740",
        local_dir_name="Machine Predictive Maintenance Classification",
    ),
    "predictive-maintenance-engine-data": DatasetSpec(
        key="predictive-maintenance-engine-data",
        title="Predictive Maintenance Engine Data",
        url=(
            "https://data.hammerhai.eu/en/dataset/"
            "mukherjee78-predictive-maintenance-engine-data"
        ),
        local_dir_name="Predictive Maintenance Engine Data",
    ),
    "smart-manufacturing-maintenance": DatasetSpec(
        key="smart-manufacturing-maintenance",
        title="Smart Manufacturing Maintenance Dataset",
        url="https://data.hammerhai.eu/en/dataset/7631074",
        local_dir_name="Smart Manufacturing Maintenance Dataset",
    ),
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
    "water-pump-rul-predictive-maintenance": DatasetSpec(
        key="water-pump-rul-predictive-maintenance",
        title="Water Pump RUL - Predictive Maintenance",
        url="https://data.hammerhai.eu/en/dataset/1832463",
        local_dir_name="Water Pump RUL - Predictive Maintenance",
    ),
}

EXAMPLE_DATASETS = MappingProxyType(_EXAMPLE_DATASETS)

_DATASET_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


def validate_dataset_key(key: str) -> str:
    """Return a key that is safe to use as one filesystem path component."""
    if not _DATASET_KEY_PATTERN.fullmatch(key):
        raise ValueError(
            "Dataset keys must be 1-80 characters, start with a letter or "
            "number, and contain only letters, numbers, dots, underscores, or "
            "hyphens."
        )
    return key


def dataset_spec_from_url(url: str) -> DatasetSpec:
    """Build a stable, filesystem-safe spec for an arbitrary DCAT-AP URL."""
    normalized_url = url.strip()
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Dataset URL must be an absolute HTTP or HTTPS URL.")
    digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:12]
    return DatasetSpec(
        key=f"url-{digest}",
        title=f"DCAT-AP dataset from {parsed.netloc}",
        url=normalized_url,
    )


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
