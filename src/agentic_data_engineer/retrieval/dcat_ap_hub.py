import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dcat_ap_hub import Dataset

from ..contracts import DatasetSpec, RetrievedDataset
from .catalog import EXAMPLE_DATASETS

DatasetFactory = Callable[[str], Any]
LocalDatasetFactory = Callable[[Path], Any]
DirectoryDatasetFactory = Callable[[Path], Any]


class DcatApHubRetriever:
    """Retrieve one of the enabled examples through dcat-ap-hub."""

    def __init__(
        self,
        *,
        dataset_factory: DatasetFactory = Dataset.from_url,
        local_dataset_factory: LocalDatasetFactory = Dataset.from_file,
        directory_dataset_factory: DirectoryDatasetFactory = Dataset.from_directory,
        verbose: bool = True,
    ) -> None:
        self._dataset_factory = dataset_factory
        self._local_dataset_factory = local_dataset_factory
        self._directory_dataset_factory = directory_dataset_factory
        self._verbose = verbose

    def retrieve(
        self,
        spec: DatasetSpec,
        destination_root: Path,
        *,
        force: bool = False,
    ) -> RetrievedDataset:
        enabled = EXAMPLE_DATASETS.get(spec.key)
        if enabled is None or enabled != spec:
            raise ValueError(
                f"Dataset {spec.key!r} is not an enabled example."
            )

        destination_root = Path(destination_root).expanduser().resolve()
        data_dir = destination_root / spec.key
        data_dir.mkdir(parents=True, exist_ok=True)

        metadata_resource = self._metadata_resource(spec)
        existing_data_dir = None if force else self._existing_dataset_dir(data_dir)
        if existing_data_dir is not None:
            dataset = self._directory_dataset_factory(existing_data_dir)
        elif metadata_resource is not None:
            dataset = self._local_dataset_factory(metadata_resource)
        else:
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
        if metadata_resource is not None:
            shutil.copyfile(
                metadata_resource,
                collection_root / "dcat-metadata.jsonld",
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

    @staticmethod
    def _existing_dataset_dir(data_dir: Path) -> Path | None:
        """Find one unambiguous existing collection without contacting its URL."""
        if (data_dir / "dcat-metadata.jsonld").is_file():
            return data_dir

        metadata_roots = sorted(
            path.parent
            for path in data_dir.glob("*/dcat-metadata.jsonld")
            if path.is_file()
        )
        if len(metadata_roots) == 1:
            return metadata_roots[0]
        if len(metadata_roots) > 1:
            roots = ", ".join(path.name for path in metadata_roots)
            raise ValueError(
                f"Multiple local dataset directories found below {data_dir}: "
                f"{roots}. Keep one collection or use --force-download."
            )

        visible_children = sorted(
            path
            for path in data_dir.iterdir()
            if not path.name.startswith(".")
        )
        if len(visible_children) == 1 and visible_children[0].is_dir():
            return visible_children[0]
        return None

    @staticmethod
    def _metadata_resource(spec: DatasetSpec) -> Path | None:
        if spec.metadata_resource is None:
            return None
        package_root = Path(__file__).resolve().parent
        resource = (package_root / spec.metadata_resource).resolve()
        if resource != package_root and package_root not in resource.parents:
            raise ValueError(
                f"Metadata resource escapes the retrieval package: {resource}."
            )
        if not resource.is_file():
            raise FileNotFoundError(
                f"Dataset metadata resource does not exist: {resource}."
            )
        return resource
