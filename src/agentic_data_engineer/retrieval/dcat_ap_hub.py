from collections.abc import Callable
from pathlib import Path
from typing import Any

from dcat_ap_hub import Dataset
from dcat_ap_hub.metadata.constants import METADATA_FILENAME, SOURCES_MANIFEST

from ..contracts import DatasetSpec, RetrievedDataset
from .catalog import validate_dataset_key

DatasetFactory = Callable[[str], Any]
LocalDatasetFactory = Callable[[Path], Any]


class DcatApHubRetriever:
    """Retrieve a DCAT-AP dataset through dcat-ap-hub."""

    def __init__(
        self,
        *,
        dataset_factory: DatasetFactory = Dataset.from_url,
        local_dataset_factory: LocalDatasetFactory = Dataset.from_directory,
        verbose: bool = True,
    ) -> None:
        self._dataset_factory = dataset_factory
        self._local_dataset_factory = local_dataset_factory
        self._verbose = verbose

    def retrieve(
        self,
        spec: DatasetSpec,
        destination_root: Path,
        *,
        force: bool = False,
    ) -> RetrievedDataset:
        destination_root = Path(destination_root).expanduser().resolve()
        validate_dataset_key(spec.key)
        if spec.local_dir_name:
            data_dir = (destination_root / spec.local_dir_name).resolve()
            if data_dir.parent != destination_root or not data_dir.is_dir():
                raise ValueError(
                    f"Configured local dataset directory does not exist: {data_dir}"
                )
            dataset = self._local_dataset_factory(data_dir)
            collection = dataset.download(
                data_dir=data_dir,
                force=False,
                verbose=self._verbose,
            )
        else:
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
        # The hub's sidecars are not data and must not mask failed downloads.
        downloaded_assets = tuple(
            path for path in files if path.name not in {METADATA_FILENAME, SOURCES_MANIFEST}
        )
        if downloaded_assets and all(
            path.suffix.lower() in {".htm", ".html"}
            for path in downloaded_assets
        ):
            names = ", ".join(path.name for path in downloaded_assets)
            raise ValueError(
                "dcat-ap-hub downloaded only HTML files instead of dataset "
                f"assets ({names}). The remote distribution may require "
                "authentication or may not expose a direct download URL."
            )
        metadata_title = str(getattr(dataset, "title", "") or spec.title)

        return RetrievedDataset(
            spec=spec,
            data_dir=collection_root,
            metadata_title=metadata_title,
            files=files,
        )
