import json
from pathlib import Path
import pandas as pd
import mlcroissant as mlc

NUMERIC_CROISSANT_TYPES = {"sc:Float", "sc:Integer"}
DATETIME_CROISSANT_TYPES = {"sc:DateTime", "sc:Date", "sc:Time"}


#TODO: split tests schreiben -> Data Leakage ..
#TODO: manuell train/test, dann über croissant einlesen -> gucken ob gleich 

# Parser/DataFrame tests

def test_timestamps(df, target_freq):
    """Prueft, ob alle Zeitabstaende im DataFrame exakt der erwarteten Frequenz entsprechen."""
    time_diffs = df.index.to_series().diff().dropna()
    if not all(time_diffs == target_freq):
        raise ValueError("Frequenz-Treue-Test fehlgeschlagen: Zeitabstände stimmen nicht überein.")


def test_numeric_columns(df):
    """Prueft, ob alle DataFrame-Spalten numerisch sind."""
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(f"Strikte Typen-Sicherheit-Test fehlgeschlagen: Nicht-numerische Spalten gefunden: {non_numeric_cols.tolist()}")


def test_datetimeindex_consistency(df):
    """Prueft, ob der DatetimeIndex zeitzonen-naiv ist."""
    if df.index.tz is not None:
        raise ValueError("Zeitzonen-Konsistenz-Test fehlgeschlagen: DatetimeIndex ist nicht zeitzonen-naiv.")


def test_no_missing_values(df):
    """Prueft, ob im finalen DataFrame keine fehlenden Werte vorkommen."""
    if df.isnull().values.any():
        raise ValueError("Vollständigkeits-Test fehlgeschlagen: Fehlende Werte im DataFrame gefunden.")


# Croissant helper functions


def _load_croissant(json_path: str | Path) -> tuple[Path, dict]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return path, json.load(f)


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
    return {rs.get("name") for rs in _record_sets(metadata) if rs.get("name")}


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


def _split_csv_path(json_path: Path, split_name: str) -> Path:
    return json_path.parent / f"{split_name}.csv"


def _csv_columns(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise ValueError(f"CSV-Datei fehlt: {csv_path}")
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


# Croissant format and schema tests


def test_croissant_format(json_path: str):
    """Prueft mit mlcroissant, ob die JSON-Datei grundsaetzlich dem Croissant-Standard folgt."""
    try:
        #TODO
        dataset = mlc.Dataset(json_path)
        if not dataset.metadata.record_sets:
            raise ValueError("In der Croissant-Datei wurden keine 'recordSet' (wie train oder test) gefunden.")
    except Exception as e:
        raise ValueError(f"Croissant-Format-Test fehlgeschlagen: Die Datei entspricht nicht dem Standard.\nDetails: {str(e)}")


def test_croissant_data_binding(json_path: str, record_set_name: str = "train"):
    """
    Testet, ob mlcroissant die tatsächlichen Daten (CSVs) anhand der
    generierten JSON-Spezifikation fehlerfrei einlesen und mappen kann.
    """
    try:
        dataset = mlc.Dataset(json_path)
        records = dataset.records(record_set=record_set_name)
        first_row = next(iter(records))
        if not first_row:
            raise ValueError("Die CSV-Datei scheint leer zu sein.")
    except StopIteration:
        raise ValueError(f"Datensatz-Test fehlgeschlagen: Das RecordSet '{record_set_name}' ist leer.")
    except FileNotFoundError:
        raise ValueError("Datensatz-Test fehlgeschlagen: Eine im JSON deklarierte Datei existiert nicht auf der Festplatte.")
    except Exception as e:
        raise ValueError(f"Datensatz-Test fehlgeschlagen: Diskrepanz zwischen JSON und CSV.\nDetails: {str(e)}")


def test_croissant_has_train_and_test(json_path: str):
    """Prueft, ob die Croissant-Datei RecordSets fuer train und test enthaelt."""
    _path, metadata = _load_croissant(json_path)
    missing = {"train", "test"} - _record_set_names(metadata)
    if missing:
        raise ValueError(f"Croissant-Test fehlgeschlagen: RecordSets fehlen: {sorted(missing)}")


def test_croissant_distribution_references_train_and_test(json_path: str):
    """Prueft, ob train.csv und test.csv in der Croissant-distribution referenziert werden."""
    _path, metadata = _load_croissant(json_path)
    referenced_files = _distribution_file_names(metadata)
    missing = {"train.csv", "test.csv"} - referenced_files
    if missing:
        raise ValueError(f"Croissant-Test fehlgeschlagen: Dateien fehlen in distribution: {sorted(missing)}")


def test_croissant_no_placeholder_hashes(json_path: str):
    """Prueft, ob alle distribution-Dateien echte sha256-Werte statt Platzhalter besitzen."""
    _path, metadata = _load_croissant(json_path)
    placeholders = []
    for item in _distribution_entries(metadata):
        sha256 = str(item.get("sha256", "")).strip().lower()
        if sha256 in {"", "placeholder", "todo", "tbd", "none"}:
            placeholders.append(item.get("name") or item.get("contentUrl") or item.get("@id") or "<unknown>")
    if placeholders:
        raise ValueError(f"Croissant-Test fehlgeschlagen: Fehlende oder Platzhalter-sha256 Werte: {placeholders}")


# CSV-to-Croissant consistency tests


def test_croissant_csv_columns_are_declared(json_path: str):
    """Prueft, ob jede Spalte aus train.csv und test.csv als Field im passenden RecordSet beschrieben ist."""
    path, metadata = _load_croissant(json_path)
    for split_name in ("train", "test"):
        record_set = _record_set_by_name(metadata, split_name)
        if record_set is None:
            raise ValueError(f"Croissant-Test fehlgeschlagen: RecordSet fehlt: {split_name}")

        csv_cols = set(_csv_columns(_split_csv_path(path, split_name)))
        field_names = set(_field_map(record_set))
        missing = csv_cols - field_names
        if missing:
            raise ValueError(
                f"Croissant-Test fehlgeschlagen: CSV-Spalten fehlen im RecordSet {split_name!r}: {sorted(missing)}"
            )


def test_croissant_has_no_extra_fields(json_path: str):
    """Prueft, ob Croissant keine Fields beschreibt, die in der jeweiligen CSV fehlen."""
    path, metadata = _load_croissant(json_path)
    for split_name in ("train", "test"):
        record_set = _record_set_by_name(metadata, split_name)
        if record_set is None:
            raise ValueError(f"Croissant-Test fehlgeschlagen: RecordSet fehlt: {split_name}")

        csv_cols = set(_csv_columns(_split_csv_path(path, split_name)))
        field_names = set(_field_map(record_set))
        extra = field_names - csv_cols
        if extra:
            raise ValueError(
                f"Croissant-Test fehlgeschlagen: Fields ohne CSV-Spalte im RecordSet {split_name!r}: {sorted(extra)}"
            )


def test_croissant_time_is_datetime(json_path: str, time_col: str = "Time"):
    """Prueft, ob die Zeitspalte in train und test als DateTime/Date/Time typisiert ist."""
    _path, metadata = _load_croissant(json_path)
    for split_name in ("train", "test"):
        record_set = _record_set_by_name(metadata, split_name)
        if record_set is None:
            raise ValueError(f"Croissant-Test fehlgeschlagen: RecordSet fehlt: {split_name}")

        field = _field_map(record_set).get(time_col)
        if field is None:
            raise ValueError(f"Croissant-Test fehlgeschlagen: Zeitspalte {time_col!r} fehlt in {split_name!r}.")

        data_type = field.get("dataType")
        if data_type not in DATETIME_CROISSANT_TYPES:
            raise ValueError(
                f"Croissant-Test fehlgeschlagen: {time_col!r} in {split_name!r} ist {data_type!r}, erwartet DateTime."
            )


def test_croissant_numeric_columns_have_numeric_types(json_path: str, time_col: str = "Time"):
    """Prueft, ob numerische CSV-Spalten im Croissant als sc:Float oder sc:Integer typisiert sind."""
    path, metadata = _load_croissant(json_path)
    for split_name in ("train", "test"):
        record_set = _record_set_by_name(metadata, split_name)
        if record_set is None:
            raise ValueError(f"Croissant-Test fehlgeschlagen: RecordSet fehlt: {split_name}")

        sample = pd.read_csv(_split_csv_path(path, split_name), nrows=1000)
        fields = _field_map(record_set)
        bad_types = []

        for col in sample.select_dtypes(include=["number"]).columns:
            if col == time_col:
                continue
            field = fields.get(col)
            if field is None:
                bad_types.append((col, "missing field"))
                continue
            data_type = field.get("dataType")
            if data_type not in NUMERIC_CROISSANT_TYPES:
                bad_types.append((col, data_type))

        if bad_types:
            raise ValueError(
                f"Croissant-Test fehlgeschlagen: Numerische CSV-Spalten haben falsche Croissant-Typen in {split_name!r}: {bad_types}"
            )


def test_croissant_all(json_path: str):
    """Fuehrt alle Croissant-Format-, Binding- und Schema-Konsistenztests aus."""
    test_croissant_format(json_path)
    test_croissant_has_train_and_test(json_path)
    test_croissant_distribution_references_train_and_test(json_path)
    test_croissant_no_placeholder_hashes(json_path)
    test_croissant_csv_columns_are_declared(json_path)
    test_croissant_has_no_extra_fields(json_path)
    test_croissant_time_is_datetime(json_path)
    test_croissant_numeric_columns_have_numeric_types(json_path)
    test_croissant_data_binding(json_path, "train")
    test_croissant_data_binding(json_path, "test")
