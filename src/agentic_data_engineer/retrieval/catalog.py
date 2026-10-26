import hashlib
import os
import re
from types import MappingProxyType
from urllib.parse import urlparse

from ..contracts import DatasetSpec

# Records served by the harvesting scripts (`uv run harvest serve` in
# harvesting-scripts) from the repaired catalogs, until they are on
# data.hammerhai.eu. Set ADE_CATALOG_URL to use another server. dcat-ap-hub
# downloads each example into data/<key>/ on its first run.
CATALOG_URL = os.environ.get("ADE_CATALOG_URL", "http://127.0.0.1:8000").rstrip("/")

_EXAMPLE_PATHS = {
    "comprehensive-multi-sheet-safety-simulation": (
        (
            "Comprehensive Multi-Sheet Simulation and Diagnostic Dataset for "
            "AI-Driven Safety Analysis in Complex Engineering Systems"
        ),
        "zenodo/10-5281-zenodo-18204207",
    ),
    "part-separator-computation-time": (
        (
            "Computation time measurements of the part separator submodel in a "
            "real-time co-simulation"
        ),
        "darus/10-18419-darus-6472",
    ),
    "edge-ai-industrial-robot-motion": (
        "Edge-AI Industrial Robot Motion Dataset",
        "kaggle/ziya07-edge-ai-industrial-robot-motion-dataset",
    ),
    "industrial-digital-twin-1-year": (
        "Industrial Digital Twin Dataset 1 Year",
        "kaggle/aiwithcagri-industrial-digital-twin-dataset-1-year",
    ),
    "industrial-equipment-sensor-anomaly": (
        "Industrial Equipment Sensor Anomaly Data",
        "huggingface/petsteb-industrial-sensor-anomaly-data",
    ),
    "industrial-machine-predictive-maintenance": (
        "Industrial Machine Predictive Maintenance (syn)",
        "kaggle/tatheerabbas-industrial-machine-predictive-maintenance",
    ),
    "industrial-screw-driving": (
        (
            "Industrial screw driving dataset collection: Time series data for "
            "process monitoring and anomaly detection"
        ),
        "zenodo/10-5281-zenodo-16031381",
    ),
    "industrial-smart-factory-automation-2026": (
        "Industrial Smart Factory Automation Dataset 2026",
        (
            "kaggle/rauffauzanrambe-industrial-smart-factory-automation-dataset-2026"
        ),
    ),
    "imaks-knowledge-extraction": (
        "iMAKS (industrial ModulAr Knowledge extraction Synthetic dataset)",
        "zenodo/10-5281-zenodo-20075430",
    ),
    "machine-failure": (
        "Machine Failure Dataset",
        "kaggle/sumit913-machine-failure-dataset",
    ),
    "predictive-maintenance-10k-synthetic-sensors": (
        "Predictive Maintenance: 10k Synthetic Sensors",
        (
            "kaggle/fernandopipaalves-predictive-maintenance-10k-synthetic-sensors"
        ),
    ),
    "predictive-maintenance-engine-data": (
        "Predictive Maintenance Engine Data",
        "huggingface/rynaxzion-predictive-maintenance-engine-data",
    ),
    "rt-fdd-pick-and-place-electric-furnace": (
        (
            "RT-FDD Datasets: Pick-and-Place and Electric Furnace Benchmarks "
            "for Real-Time Fault Detection and Diagnosis"
        ),
        "zenodo/10-5281-zenodo-20576461",
    ),
}

_EXAMPLE_DATASETS = {
    key: DatasetSpec(key=key, title=title, url=f"{CATALOG_URL}/{path}")
    for key, (title, path) in _EXAMPLE_PATHS.items()
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
