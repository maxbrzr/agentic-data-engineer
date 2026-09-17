"""Croissant Baker handler for common audio file formats."""

from pathlib import Path

import mlcroissant as mlc
from croissant_baker.handlers.base_handler import FileTypeHandler
from croissant_baker.handlers.utils import compute_file_hash


_MIME_TYPES = {
    ".aac": "audio/aac",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
}


def _has_audio_magic(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(12)
    except OSError:
        return False

    suffix = path.suffix.lower()
    if suffix == ".wav":
        return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    if suffix == ".flac":
        return header.startswith(b"fLaC")
    if suffix == ".mp3":
        return header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
        )
    if suffix in {".ogg", ".opus"}:
        return header.startswith(b"OggS")
    if suffix == ".m4a":
        return len(header) >= 8 and header[4:8] == b"ftyp"
    if suffix == ".aac":
        return len(header) >= 2 and header[0] == 0xFF and header[1] & 0xF6 == 0xF0
    if suffix in {".aif", ".aiff"}:
        return header[:4] == b"FORM" and header[8:12] in {b"AIFF", b"AIFC"}
    return False


class AudioHandler(FileTypeHandler):
    """Represent audio files as Croissant FileObjects and sc:AudioObject data."""

    EXTENSIONS = tuple(_MIME_TYPES)
    FORMAT_NAME = "Audio"
    FORMAT_DESCRIPTION = "Audio content, encoding format, size, and SHA-256"

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in _MIME_TYPES and _has_audio_magic(file_path)

    def extract_metadata(self, file_path: Path, **kwargs: object) -> dict:
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        suffix = file_path.suffix.lower()
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "sha256": compute_file_hash(file_path),
            "encoding_format": _MIME_TYPES[suffix],
        }

    def build_croissant(self, file_metas: list[dict], file_ids: list[str]) -> tuple:
        del file_ids
        formats = sorted({meta["encoding_format"] for meta in file_metas})
        extensions = sorted(
            {Path(meta["file_name"]).suffix.lower() for meta in file_metas}
        )
        count = len(file_metas)
        file_set_id = "audio-files"
        file_set = mlc.FileSet(
            id=file_set_id,
            name="Audio files",
            description=f"{count} audio files",
            encoding_formats=formats,
            includes=[f"**/*{suffix}" for suffix in extensions],
        )
        record_set = mlc.RecordSet(
            id="audio",
            name="audio",
            description=f"{count} audio files",
            fields=[
                mlc.Field(
                    id="audio/audio_content",
                    name="audio",
                    description="Audio content",
                    data_types=["sc:AudioObject"],
                    source=mlc.Source(
                        file_set=file_set_id,
                        extract=mlc.Extract(file_property="content"),
                    ),
                )
            ],
        )
        return [file_set], [record_set]
