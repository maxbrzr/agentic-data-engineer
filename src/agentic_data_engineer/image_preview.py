from pathlib import Path
from textwrap import wrap

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


_REQUIRED_AUDIT_COLUMNS = {
    "file_name",
    "original_file_name",
    "technique",
}
_PARAMETER_COLUMNS = (
    "brightness_factor",
    "contrast_factor",
    "rotation_degrees",
    "noise_std",
    "blur_radius",
)


def _resolve_preview_asset(output_dir: Path, relative_name: object) -> Path:
    if not isinstance(relative_name, str) or not relative_name.strip():
        raise ValueError("Augmentation preview paths must be non-empty strings.")
    relative_path = Path(relative_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe augmentation preview path: {relative_name!r}.")
    try:
        path = (output_dir / relative_path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Augmentation preview asset does not exist: {relative_name!r}."
        ) from exc
    root = output_dir.resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(
            f"Augmentation preview asset must be a file below output_dir: "
            f"{relative_name!r}."
        )
    return path


def _preview_tile(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        display = source.convert("RGBA")
    contained = ImageOps.contain(display, size, Image.Resampling.LANCZOS)
    tile = Image.new("RGBA", size, "white")
    left = (size[0] - contained.width) // 2
    top = (size[1] - contained.height) // 2
    tile.alpha_composite(contained, (left, top))
    return tile.convert("RGB")


def _derivative_caption(row: pd.Series) -> str:
    parts = [str(row["technique"])]
    for column in _PARAMETER_COLUMNS:
        if column not in row.index or pd.isna(row[column]) or row[column] == "":
            continue
        parts.append(f"{column}={row[column]}")
    return "\n".join(parts)


def create_augmentation_preview(
    output_dir: str | Path,
    *,
    max_originals: int = 1,
    thumbnail_size: tuple[int, int] = (224, 224),
) -> Path:
    """Create a deterministic preview of one original and its derivatives by default.

    ``augmentation.csv`` supplies all paths and captions. Source images are
    converted to RGBA only for display; exported training assets are untouched.
    The resulting PNG is written atomically to ``augmentation_preview.png``.
    """
    root = Path(output_dir)
    audit_path = root / "augmentation.csv"
    if not audit_path.is_file():
        raise ValueError(f"Augmentation audit CSV is missing: {audit_path}.")
    if max_originals < 1:
        raise ValueError("max_originals must be at least 1.")
    if len(thumbnail_size) != 2 or min(thumbnail_size) < 1:
        raise ValueError("thumbnail_size must contain two positive integers.")

    audit = pd.read_csv(audit_path)
    missing = sorted(_REQUIRED_AUDIT_COLUMNS - set(audit.columns))
    if missing:
        raise ValueError(f"augmentation.csv is missing required columns: {missing}.")
    if audit.empty:
        raise ValueError("augmentation.csv must contain at least one derivative.")

    selected_originals = sorted(audit["original_file_name"].unique())[:max_originals]
    grouped = [
        audit[audit["original_file_name"] == original]
        .sort_values(["technique", "file_name"], kind="stable")
        .reset_index(drop=True)
        for original in selected_originals
    ]
    column_count = 1 + max(len(group) for group in grouped)
    cell_width = thumbnail_size[0] + 24
    caption_height = 64
    cell_height = thumbnail_size[1] + caption_height + 16
    canvas = Image.new(
        "RGB",
        (column_count * cell_width, len(grouped) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for row_index, (original_name, derivatives) in enumerate(
        zip(selected_originals, grouped, strict=True)
    ):
        entries: list[tuple[str, str]] = [(original_name, "original")]
        entries.extend(
            (str(row["file_name"]), _derivative_caption(row))
            for _, row in derivatives.iterrows()
        )
        for column_index, (file_name, caption) in enumerate(entries):
            path = _resolve_preview_asset(root, file_name)
            tile = _preview_tile(path, thumbnail_size)
            x = column_index * cell_width + 12
            y = row_index * cell_height + 8
            canvas.paste(tile, (x, y))
            caption_lines = []
            for line in caption.splitlines() or [caption]:
                caption_lines.extend(wrap(line, width=32) or [""])
            draw.multiline_text(
                (x, y + thumbnail_size[1] + 6),
                "\n".join(caption_lines[:3]),
                fill="black",
                spacing=2,
            )

    preview_path = root / "augmentation_preview.png"
    temporary_path = root / ".augmentation_preview.tmp.png"
    try:
        canvas.save(temporary_path, format="PNG")
        temporary_path.replace(preview_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return preview_path
