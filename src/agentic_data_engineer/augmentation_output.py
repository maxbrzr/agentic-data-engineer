import csv
from pathlib import Path


def augmentation_audit_error(output_dir: Path) -> str | None:
    """Check that the augmentation audit has a header and at least one entry."""
    path = output_dir / "augmentation.csv"
    if not path.is_file():
        return "augmentation.csv is missing"
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream, strict=True)
            header = next(rows, ())
            if any(cell.strip() for cell in header) and any(
                any(cell.strip() for cell in row) for row in rows
            ):
                return None
    except (OSError, UnicodeError, csv.Error):
        return "augmentation.csv is unreadable"
    return "augmentation.csv must contain a header and at least one audit entry"
