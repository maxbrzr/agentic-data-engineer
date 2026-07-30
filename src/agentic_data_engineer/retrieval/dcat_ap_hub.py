from collections.abc import Callable
from pathlib import Path
from typing import Any

from dcat_ap_hub import Dataset

from ..contracts import DatasetSpec, RetrievedDataset
from .catalog import EXAMPLE_DATASETS

DatasetFactory = Callable[[str], Any]


class DcatApHubRetriever:
    """Retrieve one of the enabled examples through dcat-ap-hub."""

    def __init__(
        self,
        *,
        dataset_factory: DatasetFactory = Dataset.from_url,
        verbose: bool = True,
    ) -> None:
        self._dataset_factory = dataset_factory
        self._verbose = verbose

    def retrieve(
        self,
        spec: DatasetSpec,
        destination_root: Path,
        *,
        force: bool = False,
    ) -> RetrievedDataset:
        enabled = EXAMPLE_DATASETS.get(spec.key)
        if enabled is None or enabled.url != spec.url:
            raise ValueError(
                f"Dataset {spec.key!r} is not one of the three enabled examples."
            )

        destination_root = Path(destination_root).expanduser().resolve()
        data_dir = destination_root / spec.key
        data_dir.mkdir(parents=True, exist_ok=True)

        dataset = self._dataset_factory(spec.url)
        collection = dataset.download(
            data_dir=data_dir,
            force=force,
            verbose=self._verbose,
        )

        collection_root = Path(getattr(collection, "root", data_dir)).resolve()
        if collection_root != data_dir and data_dir not in collection_root.parents:
            raise ValueError(
                f"dcat-ap-hub returned a data path outside the dataset directory: "
                f"{collection_root}"
            )
        files = tuple(
            sorted(
                (
                    Path(asset.path).resolve()
                    for asset in collection
                    if getattr(asset, "path", None) is not None
                ),
                key=str,
            )
        )
        metadata_title = str(getattr(dataset, "title", "") or spec.title)

        return RetrievedDataset(
            spec=spec,
            data_dir=collection_root,
            metadata_title=metadata_title,
            files=files,
        )
