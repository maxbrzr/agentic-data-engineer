import hashlib
import re
from types import MappingProxyType
from urllib.parse import urlparse

from ..contracts import DatasetSpec

# Examples backed by folders already downloaded under data/. The URL is the
# dataset's record on data.hammerhai.eu; Zenodo records are not published there
# yet, so those use the DOI. Local examples never fetch their URL.
_EXAMPLE_DATASETS = {
    "comprehensive-multi-sheet-safety-simulation": DatasetSpec(
        key="comprehensive-multi-sheet-safety-simulation",
        title=(
            "Comprehensive Multi-Sheet Simulation and Diagnostic Dataset for "
            "AI-Driven Safety Analysis in Complex Engineering Systems"
        ),
        url="https://doi.org/10.5281/zenodo.18204207",
        local_dir_name=(
            "Comprehensive Multi-Sheet Simulation and Diagnostic Dataset for "
            "AI-Driven Safety Analysis in Complex Engineering Systems"
        ),
    ),
    "part-separator-computation-time": DatasetSpec(
        key="part-separator-computation-time",
        title=(
            "Computation time measurements of the part separator submodel in a "
            "real-time co-simulation"
        ),
        url="https://data.hammerhai.eu/en/dataset/doi-10-18419-darus-6472",
        local_dir_name=(
            "Computation time measurements of the part separator submodel in a "
            "real-time co-simulation"
        ),
    ),
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
    "industrial-equipment-sensor-anomaly": DatasetSpec(
        key="industrial-equipment-sensor-anomaly",
        title="Industrial Equipment Sensor Anomaly Data",
        url=(
            "https://data.hammerhai.eu/en/dataset/"
            "petsteb-industrial-sensor-anomaly-data"
        ),
        local_dir_name="Industrial Equipment Sensor Anomaly Data",
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
    "industrial-screw-driving": DatasetSpec(
        key="industrial-screw-driving",
        title=(
            "Industrial screw driving dataset collection: Time series data for "
            "process monitoring and anomaly detection"
        ),
        url="https://doi.org/10.5281/zenodo.16031381",
        local_dir_name=(
            "Industrial screw driving dataset collection Time series data for "
            "process monitoring and anomaly detection"
        ),
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
    "imaks-knowledge-extraction": DatasetSpec(
        key="imaks-knowledge-extraction",
        title="iMAKS (industrial ModulAr Knowledge extraction Synthetic dataset)",
        url="https://doi.org/10.5281/zenodo.20075430",
        local_dir_name=(
            "iMAKS (industrial ModulAr Knowledge extraction Synthetic dataset)"
        ),
    ),
    "machine-failure": DatasetSpec(
        key="machine-failure",
        title="Machine Failure Dataset",
        url="https://data.hammerhai.eu/en/dataset/sumit913-machine-failure-dataset",
        local_dir_name="Machine Failure Dataset",
    ),
    "predictive-maintenance-10k-synthetic-sensors": DatasetSpec(
        key="predictive-maintenance-10k-synthetic-sensors",
        title="Predictive Maintenance: 10k Synthetic Sensors",
        url=(
            "https://data.hammerhai.eu/en/dataset/"
            "fernandopipaalves-predictive-maintenance-10k-synthetic-sensors"
        ),
        local_dir_name="Predictive Maintenance 10k Synthetic Sensors",
    ),
    "predictive-maintenance-engine-data": DatasetSpec(
        key="predictive-maintenance-engine-data",
        title="Predictive Maintenance Engine Data",
        url=(
            "https://data.hammerhai.eu/en/dataset/"
            "rynaxzion-predictive-maintenance-engine-data"
        ),
        local_dir_name="Predictive Maintenance Engine Data",
    ),
    "rt-fdd-pick-and-place-electric-furnace": DatasetSpec(
        key="rt-fdd-pick-and-place-electric-furnace",
        title=(
            "RT-FDD Datasets: Pick-and-Place and Electric Furnace Benchmarks "
            "for Real-Time Fault Detection and Diagnosis"
        ),
        url="https://doi.org/10.5281/zenodo.20576461",
        local_dir_name=(
            "RT-FDD Datasets Pick-and-Place and Electric Furnace Benchmarks "
            "for Real-Time Fault Detection and Diagnosis"
        ),
    ),
    "time-series-data": DatasetSpec(
        key="time-series-data",
        title="Time Series Data",
        url="https://data.hammerhai.eu/en/dataset/deeeja-time-series-data",
        local_dir_name="Time Series Data",
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
