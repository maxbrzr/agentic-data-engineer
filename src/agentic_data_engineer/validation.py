import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

NUMERIC_CROISSANT_TYPES = {
    "sc:Float",
    "sc:Integer",
    "cr:Float16",
    "cr:Float32",
    "cr:Float64",
    "cr:Int8",
    "cr:Int16",
    "cr:Int32",
    "cr:Int64",
    "cr:UInt8",
    "cr:UInt16",
    "cr:UInt32",
    "cr:UInt64",
}
DATETIME_CROISSANT_TYPES = {"sc:DateTime", "sc:Date", "sc:Time"}


# DataFrame validation


def _require_dataframe(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(df).__name__}.")


def _require_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataframe_name: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataframe_name} is missing required columns: {missing}.")


def test_non_empty(df: pd.DataFrame, dataframe_name: str = "DataFrame") -> None:
    """Require at least one row and one column."""
    _require_dataframe(df, dataframe_name)
    if df.empty or len(df.columns) == 0:
        raise ValueError(
            f"{dataframe_name} must contain at least one row and one column."
        )


def test_unique_columns(df: pd.DataFrame, dataframe_name: str = "DataFrame") -> None:
    """Require unique column names."""
    _require_dataframe(df, dataframe_name)
    duplicates = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            f"{dataframe_name} contains duplicate column names: {duplicates}."
        )


def test_numeric_columns(
    df: pd.DataFrame,
    exclude_columns: Sequence[str] | None = None,
) -> None:
    """Require model columns to be numeric or boolean."""
    _require_dataframe(df, "DataFrame")
    excluded = set(exclude_columns or ())
    unknown_exclusions = sorted(excluded - set(df.columns))
    if unknown_exclusions:
        raise ValueError(
            f"Excluded columns are not present in the DataFrame: {unknown_exclusions}."
        )

    non_numeric = [
        column
        for column in df.columns
        if column not in excluded
        and not (is_numeric_dtype(df[column].dtype) or is_bool_dtype(df[column].dtype))
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric model columns found: {non_numeric}.")


def test_no_missing_values(df: pd.DataFrame) -> None:
    """Require every value to be present."""
    _require_dataframe(df, "DataFrame")
    missing_counts = df.isna().sum()
    missing = {
        str(column): int(count) for column, count in missing_counts.items() if count > 0
    }
    if missing:
        raise ValueError(f"Missing values found: {missing}.")


def test_no_infinite_values(
    df: pd.DataFrame,
    exclude_columns: Sequence[str] | None = None,
) -> None:
    """Reject positive and negative infinity in numeric model columns."""
    _require_dataframe(df, "DataFrame")
    excluded = set(exclude_columns or ())
    infinite_counts: dict[str, int] = {}

    for column in df.columns:
        if column in excluded or not is_numeric_dtype(df[column].dtype):
            continue
        count = int(np.isinf(df[column].to_numpy(dtype=float, na_value=np.nan)).sum())
        if count:
            infinite_counts[str(column)] = count

    if infinite_counts:
        raise ValueError(f"Infinite values found: {infinite_counts}.")


def test_target_column(
    df: pd.DataFrame,
    target_col: str,
    *,
    require_non_constant: bool = True,
    dataframe_name: str = "DataFrame",
) -> None:
    """Require a present, complete, and optionally non-constant target."""
    _require_dataframe(df, dataframe_name)
    _require_columns(df, [target_col], dataframe_name=dataframe_name)

    target = df[target_col]
    if target.isna().any():
        raise ValueError(
            f"Target column {target_col!r} contains missing values in {dataframe_name}."
        )
    if require_non_constant and target.nunique(dropna=False) < 2:
        raise ValueError(
            f"Target column {target_col!r} is constant in {dataframe_name}."
        )


def test_timestamps(df: pd.DataFrame, expected_freq: str | pd.Timedelta) -> None:
    """Require a sorted, unique DatetimeIndex with one exact interval."""
    _require_dataframe(df, "DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Timestamp validation requires a pandas DatetimeIndex.")
    if len(df.index) < 2:
        raise ValueError(
            "At least two timestamps are required to validate a frequency."
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be sorted in increasing order.")
    if not df.index.is_unique:
        raise ValueError("DatetimeIndex contains duplicate timestamps.")

    try:
        expected = pd.Timedelta(expected_freq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid expected frequency {expected_freq!r}.") from exc
    if expected <= pd.Timedelta(0):
        raise ValueError("Expected frequency must be greater than zero.")

    time_diffs = df.index.to_series().diff().dropna()
    mismatches = int((time_diffs != expected).sum())
    if mismatches:
        observed = sorted({str(value) for value in time_diffs.unique()})
        raise ValueError(
            f"Timestamp frequency mismatch in {mismatches} interval(s); "
            f"expected {expected}, observed {observed[:10]}."
        )


def test_datetimeindex_consistency(
    df: pd.DataFrame,
    expected_timezone: str | None = None,
) -> None:
    """Require a DatetimeIndex with the expected timezone convention."""
    _require_dataframe(df, "DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a pandas DatetimeIndex.")

    actual_timezone = None if df.index.tz is None else str(df.index.tz)
    if actual_timezone != expected_timezone:
        raise ValueError(
            f"DatetimeIndex timezone is {actual_timezone!r}; "
            f"expected {expected_timezone!r}."
        )


# Split and leakage validation


def test_split_schema(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Require identical ordered columns and dtypes in both splits."""
    test_non_empty(train_df, "Training split")
    test_non_empty(test_df, "Test split")
    test_unique_columns(train_df, "Training split")
    test_unique_columns(test_df, "Test split")

    train_columns = list(train_df.columns)
    test_columns = list(test_df.columns)
    if train_columns != test_columns:
        raise ValueError(
            "Train/test columns differ in names or order: "
            f"train={train_columns}, test={test_columns}."
        )

    mismatched_dtypes = {
        column: (str(train_df[column].dtype), str(test_df[column].dtype))
        for column in train_columns
        if train_df[column].dtype != test_df[column].dtype
    }
    if mismatched_dtypes:
        raise ValueError(f"Train/test dtypes differ: {mismatched_dtypes}.")


def _row_hashes(df: pd.DataFrame, columns: Sequence[str]) -> pd.Index:
    _require_columns(df, columns, dataframe_name="DataFrame")
    return pd.Index(pd.util.hash_pandas_object(df[list(columns)], index=False).unique())


def test_no_shared_rows(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key_columns: Sequence[str] | None = None,
) -> None:
    """Reject observations with the same key in both splits."""
    test_split_schema(train_df, test_df)
    columns = list(key_columns) if key_columns is not None else list(train_df.columns)
    if not columns:
        raise ValueError(
            "At least one key column is required for cross-split row checks."
        )

    shared = _row_hashes(train_df, columns).intersection(_row_hashes(test_df, columns))
    if len(shared):
        raise ValueError(
            f"Found {len(shared)} shared observation key(s) across train and test "
            f"using columns {columns}."
        )


def test_no_group_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_columns: Sequence[str],
) -> None:
    """Require every protected group to occur in only one split."""
    columns = list(group_columns)
    if not columns:
        raise ValueError(
            "At least one group column is required for group-leakage validation."
        )

    train_groups = _row_hashes(train_df, columns)
    test_groups = _row_hashes(test_df, columns)
    overlap = train_groups.intersection(test_groups)
    if len(overlap):
        raise ValueError(
            f"Found {len(overlap)} protected group(s) in both splits "
            f"using columns {columns}."
        )


def _datetime_values(
    df: pd.DataFrame, time_col: str | None, split_name: str
) -> pd.Series:
    if time_col is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"{split_name} must use a DatetimeIndex when no time column is supplied."
            )
        values = pd.Series(df.index, index=df.index)
    else:
        _require_columns(df, [time_col], dataframe_name=split_name)
        values = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    if values.isna().any():
        raise ValueError(f"{split_name} contains invalid or missing timestamps.")
    return values


def test_chronological_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str | None = None,
) -> None:
    """Require all training observations to precede all test observations."""
    train_time = _datetime_values(train_df, time_col, "Training split")
    test_time = _datetime_values(test_df, time_col, "Test split")
    if train_time.max() >= test_time.min():
        raise ValueError(
            "Chronological leakage detected: the latest training timestamp must be "
            "strictly earlier than the earliest test timestamp."
        )


def test_classification_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    *,
    require_contiguous_integers: bool = True,
) -> None:
    """Validate classification label coverage and optional integer encoding."""
    test_target_column(train_df, target_col, dataframe_name="Training split")
    test_target_column(
        test_df,
        target_col,
        require_non_constant=False,
        dataframe_name="Test split",
    )

    train_labels = set(train_df[target_col].unique().tolist())
    test_labels = set(test_df[target_col].unique().tolist())
    unseen = test_labels - train_labels
    if unseen:
        raise ValueError(
            f"Test split contains labels not present in training: {sorted(unseen)}."
        )

    if require_contiguous_integers:
        target = train_df[target_col]
        if not is_integer_dtype(target.dtype) or is_bool_dtype(target.dtype):
            raise ValueError("Classification labels must use an integer dtype.")
        expected = set(range(len(train_labels)))
        if train_labels != expected:
            raise ValueError(
                f"Classification labels must be contiguous integers starting at 0; "
                f"found {sorted(train_labels)}."
            )


def test_deterministic_splits(
    first_train: pd.DataFrame,
    first_test: pd.DataFrame,
    second_train: pd.DataFrame,
    second_test: pd.DataFrame,
) -> None:
    """Require two runs to produce byte-equivalent logical split tables."""
    try:
        pd.testing.assert_frame_equal(first_train, second_train, check_exact=True)
        pd.testing.assert_frame_equal(first_test, second_test, check_exact=True)
    except AssertionError as exc:
        raise ValueError(
            f"Repeated processing produced different splits: {exc}"
        ) from exc


def test_tabular_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    *,
    task_type: str,
    exclude_from_numeric: Sequence[str] | None = None,
    key_columns: Sequence[str] | None = None,
    group_columns: Sequence[str] | None = None,
    time_col: str | None = None,
    require_chronological: bool = False,
) -> None:
    """Run the generic gatekeeper suite for final tabular train/test outputs."""
    normalized_task = task_type.strip().lower()
    if normalized_task not in {"classification", "regression"}:
        raise ValueError("task_type must be either 'classification' or 'regression'.")
    if target_col in set(exclude_from_numeric or ()):
        raise ValueError(
            "The target column cannot be excluded from numeric validation."
        )

    test_split_schema(train_df, test_df)
    for split_name, split_df in (("Training split", train_df), ("Test split", test_df)):
        test_no_missing_values(split_df)
        test_no_infinite_values(split_df, exclude_from_numeric)
        test_numeric_columns(split_df, exclude_from_numeric)
        test_target_column(
            split_df,
            target_col,
            require_non_constant=split_name == "Training split",
            dataframe_name=split_name,
        )

    test_no_shared_rows(train_df, test_df, key_columns)
    if group_columns is not None:
        test_no_group_leakage(train_df, test_df, group_columns)
    if require_chronological:
        test_chronological_split(train_df, test_df, time_col)
    if normalized_task == "classification":
        test_classification_labels(train_df, test_df, target_col)


# Croissant helpers


def _load_croissant(json_path: str | Path) -> tuple[Path, dict]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as stream:
        return path, json.load(stream)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _record_sets(metadata: dict) -> list[dict]:
    return _as_list(metadata.get("recordSet"))


def _record_set_by_name(metadata: dict, name: str) -> dict | None:
    for record_set in _record_sets(metadata):
        if record_set.get("name") == name:
            return record_set
    return None


def _record_set_names(metadata: dict) -> set[str]:
    return {
        record_set.get("name")
        for record_set in _record_sets(metadata)
        if record_set.get("name")
    }


def _field_map(record_set: dict) -> dict[str, dict]:
    return {
        field.get("name"): field
        for field in _as_list(record_set.get("field"))
        if field.get("name")
    }


def _distribution_entries(metadata: dict) -> list[dict]:
    return _as_list(metadata.get("distribution"))


def _distribution_file_names(metadata: dict) -> set[str]:
    names = set()
    for item in _distribution_entries(metadata):
        for key in ("name", "contentUrl", "@id"):
            value = item.get(key)
            if value:
                names.add(Path(str(value)).name)
    return names


def _distribution_by_id(metadata: dict) -> dict[str, dict]:
    return {
        item["@id"]: item for item in _distribution_entries(metadata) if item.get("@id")
    }


def _split_csv_path(json_path: Path, split_name: str) -> Path:
    return json_path.parent / f"{split_name}.csv"


def _csv_columns(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise ValueError(f"CSV file is missing: {csv_path}.")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        try:
            return next(csv.reader(stream))
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {csv_path}.") from exc


def _local_distribution_path(json_path: Path, item: dict) -> Path:
    content_url = str(item.get("contentUrl", "")).strip()
    if not content_url:
        raise ValueError(
            f"Distribution {item.get('@id', '<unknown>')!r} has no contentUrl."
        )
    parsed = urlparse(content_url)
    if parsed.scheme or parsed.netloc:
        raise ValueError(
            f"Generated distribution must reference a local file: {content_url!r}."
        )

    candidate = (json_path.parent / content_url).resolve()
    output_root = json_path.parent.resolve()
    if candidate != output_root and output_root not in candidate.parents:
        raise ValueError(
            f"Distribution path escapes the output directory: {content_url!r}."
        )
    return candidate


# Croissant validation


def test_croissant_format(json_path: str | Path) -> None:
    """Require metadata that the ML Croissant library can load."""
    import mlcroissant as mlc

    try:
        dataset = mlc.Dataset(json_path)
        if not dataset.metadata.record_sets:
            raise ValueError("Croissant metadata contains no record sets.")
    except Exception as exc:
        raise ValueError(f"Invalid ML Croissant metadata: {exc}") from exc


def test_croissant_data_binding(
    json_path: str | Path,
    record_set_name: str = "train",
) -> None:
    """Require ML Croissant to read at least one record from a record set."""
    import mlcroissant as mlc

    try:
        dataset = mlc.Dataset(json_path)
        first_row = next(iter(dataset.records(record_set=record_set_name)))
        if not first_row:
            raise ValueError("The bound CSV appears to be empty.")
    except StopIteration as exc:
        raise ValueError(f"Record set {record_set_name!r} is empty.") from exc
    except FileNotFoundError as exc:
        raise ValueError("A file declared by Croissant does not exist.") from exc
    except Exception as exc:
        raise ValueError(
            f"Croissant data binding failed for record set {record_set_name!r}: {exc}"
        ) from exc


def test_croissant_has_train_and_test(json_path: str | Path) -> None:
    """Require train and test record sets."""
    _path, metadata = _load_croissant(json_path)
    missing = {"train", "test"} - _record_set_names(metadata)
    if missing:
        raise ValueError(f"Croissant record sets are missing: {sorted(missing)}.")


def test_croissant_distribution_references_train_and_test(
    json_path: str | Path,
) -> None:
    """Require train.csv and test.csv distribution entries."""
    _path, metadata = _load_croissant(json_path)
    missing = {"train.csv", "test.csv"} - _distribution_file_names(metadata)
    if missing:
        raise ValueError(
            f"Croissant distributions are missing files: {sorted(missing)}."
        )


def test_croissant_file_objects(json_path: str | Path) -> None:
    """Require complete, uniquely identified local FileObject entries."""
    path, metadata = _load_croissant(json_path)
    entries = _distribution_entries(metadata)
    if not entries:
        raise ValueError("Croissant distribution is empty.")

    ids = [item.get("@id") for item in entries]
    duplicate_ids = sorted(
        {item_id for item_id in ids if item_id and ids.count(item_id) > 1}
    )
    if duplicate_ids:
        raise ValueError(
            f"Croissant distribution contains duplicate @id values: {duplicate_ids}."
        )

    required = {"@id", "name", "contentUrl", "encodingFormat", "sha256"}
    for item in entries:
        missing = sorted(key for key in required if not str(item.get(key, "")).strip())
        if item.get("@type") != "cr:FileObject":
            raise ValueError(
                f"Distribution {item.get('@id', '<unknown>')!r} is not a cr:FileObject."
            )
        if missing:
            raise ValueError(
                f"Distribution {item.get('@id', '<unknown>')!r} is missing fields: {missing}."
            )
        file_path = _local_distribution_path(path, item)
        if not file_path.is_file():
            raise ValueError(f"Distribution file does not exist: {file_path}.")


def test_croissant_hashes_match_files(json_path: str | Path) -> None:
    """Require every declared SHA-256 hash to match its local file."""
    path, metadata = _load_croissant(json_path)
    for item in _distribution_entries(metadata):
        file_path = _local_distribution_path(path, item)
        if not file_path.is_file():
            raise ValueError(f"Distribution file does not exist: {file_path}.")
        expected = str(item.get("sha256", "")).strip().lower()
        if len(expected) != 64 or any(
            char not in "0123456789abcdef" for char in expected
        ):
            raise ValueError(
                f"Invalid SHA-256 value for {file_path.name}: {expected!r}."
            )

        digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {file_path.name}: expected {expected}, actual {actual}."
            )


def test_croissant_csv_columns_are_declared(json_path: str | Path) -> None:
    """Require every CSV column to have a matching Croissant field."""
    path, metadata = _load_croissant(json_path)
    for split_name in ("train", "test"):
        record_set = _record_set_by_name(metadata, split_name)
        if record_set is None:
            raise ValueError(f"Croissant record set is missing: {split_name!r}.")

        csv_columns = _csv_columns(_split_csv_path(path, split_name))
        duplicate_columns = sorted(
            {column for column in csv_columns if csv_columns.count(column) > 1}
        )
        if duplicate_columns:
            raise ValueError(
                f"{split_name}.csv contains duplicate columns: {duplicate_columns}."
            )
        missing = set(csv_columns) - set(_field_map(record_set))
        if missing:
            raise ValueError(
                f"CSV columns are missing from record set {split_name!r}: {sorted(missing)}."
            )


def test_croissant_has_no_extra_fields(json_path: str | Path) -> None:
    """Reject Croissant fields that do not exist in the corresponding CSV."""
    path, metadata = _load_croissant(json_path)
    for split_name in ("train", "test"):
        record_set = _record_set_by_name(metadata, split_name)
        if record_set is None:
            raise ValueError(f"Croissant record set is missing: {split_name!r}.")

        csv_columns = set(_csv_columns(_split_csv_path(path, split_name)))
        extra = set(_field_map(record_set)) - csv_columns
        if extra:
            raise ValueError(
                f"Croissant fields have no CSV column in {split_name!r}: {sorted(extra)}."
            )


def test_croissant_field_sources(json_path: str | Path) -> None:
    """Require every field to bind its exact CSV column and matching FileObject."""
    path, metadata = _load_croissant(json_path)
    distributions = _distribution_by_id(metadata)

    for split_name in ("train", "test"):
        record_set = _record_set_by_name(metadata, split_name)
        if record_set is None:
            raise ValueError(f"Croissant record set is missing: {split_name!r}.")
        expected_file = f"{split_name}.csv"

        for field_name, field in _field_map(record_set).items():
            source = field.get("source")
            if not isinstance(source, dict):
                raise ValueError(
                    f"Field {field_name!r} in {split_name!r} has no valid source object."
                )
            file_object = source.get("fileObject")
            extract = source.get("extract")
            file_id = file_object.get("@id") if isinstance(file_object, dict) else None
            column = extract.get("column") if isinstance(extract, dict) else None
            if file_id not in distributions:
                raise ValueError(
                    f"Field {field_name!r} references unknown FileObject {file_id!r}."
                )
            referenced_path = _local_distribution_path(path, distributions[file_id])
            if referenced_path.name != expected_file:
                raise ValueError(
                    f"Field {field_name!r} in {split_name!r} references "
                    f"{referenced_path.name!r}, expected {expected_file!r}."
                )
            if column != field_name:
                raise ValueError(
                    f"Field {field_name!r} extracts column {column!r}; "
                    "the names must match exactly."
                )


def test_croissant_time_is_datetime(
    json_path: str | Path,
    time_col: str,
) -> None:
    """Require an explicitly selected time field to use a datetime Croissant type."""
    _path, metadata = _load_croissant(json_path)
    for split_name in ("train", "test"):
        record_set = _record_set_by_name(metadata, split_name)
        if record_set is None:
            raise ValueError(f"Croissant record set is missing: {split_name!r}.")

        field = _field_map(record_set).get(time_col)
        if field is None:
            raise ValueError(
                f"Time column {time_col!r} is missing from {split_name!r}."
            )
        data_type = field.get("dataType")
        if data_type not in DATETIME_CROISSANT_TYPES:
            raise ValueError(
                f"Time column {time_col!r} in {split_name!r} uses {data_type!r}; "
                f"expected one of {sorted(DATETIME_CROISSANT_TYPES)}."
            )


def test_croissant_numeric_columns_have_numeric_types(
    json_path: str | Path,
    time_col: str | None = None,
) -> None:
    """Require numeric CSV columns to use numeric Croissant types."""
    path, metadata = _load_croissant(json_path)
    for split_name in ("train", "test"):
        record_set = _record_set_by_name(metadata, split_name)
        if record_set is None:
            raise ValueError(f"Croissant record set is missing: {split_name!r}.")

        sample = pd.read_csv(_split_csv_path(path, split_name), nrows=1000)
        fields = _field_map(record_set)
        bad_types = []
        for column in sample.select_dtypes(include=["number"]).columns:
            if time_col is not None and column == time_col:
                continue
            field = fields.get(column)
            data_type = None if field is None else field.get("dataType")
            if data_type not in NUMERIC_CROISSANT_TYPES:
                bad_types.append((column, data_type or "missing field"))

        if bad_types:
            raise ValueError(
                f"Numeric CSV columns use invalid Croissant types in "
                f"{split_name!r}: {bad_types}."
            )


def test_croissant_all(
    json_path: str | Path,
    time_col: str | None = None,
) -> None:
    """Run all generic Croissant checks and optionally validate a datetime column."""
    test_croissant_format(json_path)
    test_croissant_has_train_and_test(json_path)
    test_croissant_distribution_references_train_and_test(json_path)
    test_croissant_file_objects(json_path)
    test_croissant_hashes_match_files(json_path)
    test_croissant_csv_columns_are_declared(json_path)
    test_croissant_has_no_extra_fields(json_path)
    test_croissant_field_sources(json_path)
    if time_col is not None:
        test_croissant_time_is_datetime(json_path, time_col)
    test_croissant_numeric_columns_have_numeric_types(json_path, time_col)
    test_croissant_data_binding(json_path, "train")
    test_croissant_data_binding(json_path, "test")
