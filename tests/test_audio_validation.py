import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_data_engineer.validation import (
    test_audio_validation as validate_audio_output,
)


class AudioValidationTests(unittest.TestCase):
    """Validate audio exports using small PCM WAV recordings."""

    def _write_audio(self, relative_path: str, frequency: int) -> None:
        # A short mono sine wave with 16-bit samples at 8 kHz.
        time = np.arange(800) / 8000
        samples = (10000 * np.sin(2 * np.pi * frequency * time)).astype("<i2")
        with wave.open(str(self.output_dir / relative_path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            stream.writeframes(samples.tobytes())

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary_directory.name)
        (self.output_dir / "train" / "audio").mkdir(parents=True)
        (self.output_dir / "test" / "audio").mkdir(parents=True)

        self._write_audio("train/audio/tone_440.wav", 440)
        self._write_audio("train/audio/tone_660.wav", 660)
        self._write_audio("test/audio/tone_880.wav", 880)
        pd.DataFrame(
            {
                "file_name": [
                    "train/audio/tone_440.wav",
                    "train/audio/tone_660.wav",
                ],
                "label": ["scratch", "clean"],
            }
        ).to_csv(self.output_dir / "train.csv", index=False)
        pd.DataFrame(
            {
                "file_name": ["test/audio/tone_880.wav"],
                "label": ["scratch"],
            }
        ).to_csv(self.output_dir / "test.csv", index=False)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_valid_audio_splits_pass(self):
        validate_audio_output(self.output_dir, "label")

    def test_nested_audio_path_passes(self):
        audio_dir = self.output_dir / "test" / "audio"
        (audio_dir / "batch").mkdir()
        (audio_dir / "tone_880.wav").rename(audio_dir / "batch" / "tone_880.wav")
        manifest_path = self.output_dir / "test.csv"
        manifest = pd.read_csv(manifest_path)
        manifest.loc[0, "file_name"] = "test/audio/batch/tone_880.wav"
        manifest.to_csv(manifest_path, index=False)

        validate_audio_output(self.output_dir, "label")

    def test_valid_multi_label_splits_pass_with_one_row_per_recording(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = [
            repr(["paper", "scratch"]),
            repr(["clean", "paper"]),
        ]
        test["label"] = [repr(["paper", "scratch"])]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        validate_audio_output(self.output_dir, "label", multi_label=True)
        self.assertEqual(len(pd.read_csv(self.output_dir / "train.csv")), 2)
        self.assertEqual(len(pd.read_csv(self.output_dir / "test.csv")), 1)

    def test_valid_integer_multi_label_splits_pass(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = [repr([1, 2]), repr([2, 3])]
        test["label"] = [repr([1, 3])]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        validate_audio_output(self.output_dir, "label", multi_label=True)

    def test_inline_captions_and_events_pass_without_sidecars(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["captions"] = [repr(["tone_440 sample", "steady tone"]), repr(["tone_660 sample"])]
        test["captions"] = [repr(["tone_880 sample"])]
        train["events"] = [
            repr([{"label": "tone", "start_seconds": 0.0, "end_seconds": 0.1}]),
            repr([]),
        ]
        test["events"] = [repr([])]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        validate_audio_output(
            self.output_dir,
            "label",
            list_columns=["captions", "events"],
        )
        self.assertFalse((self.output_dir / "train_captions.csv").exists())
        self.assertFalse((self.output_dir / "train_events.csv").exists())

    def test_inline_annotation_column_must_use_list_literal(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["captions"] = ["tone_440 sample", repr(["tone_660 sample"])]
        test["captions"] = [repr(["tone_880 sample"])]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        with self.assertRaisesRegex(ValueError, "invalid list literal"):
            validate_audio_output(
                self.output_dir,
                "label",
                list_columns=["captions"],
            )

    def test_multi_label_unseen_test_label_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = ["['clean', 'paper']", "['paper', 'scratch']"]
        test["label"] = ["['paper', 'unknown']"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        with self.assertRaisesRegex(ValueError, "not present in training.*unknown"):
            validate_audio_output(self.output_dir, "label", multi_label=True)

    def test_multi_label_lists_must_be_sorted_and_unique(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = ["['scratch', 'paper']", "['clean', 'paper']"]
        test["label"] = ["['paper', 'scratch']"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        with self.assertRaisesRegex(ValueError, "sort labels deterministically"):
            validate_audio_output(self.output_dir, "label", multi_label=True)

        train.loc[0, "label"] = "['paper', 'paper']"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "duplicate labels"):
            validate_audio_output(self.output_dir, "label", multi_label=True)

    def test_multi_label_cell_must_be_a_list_literal(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = ["paper,scratch", "['clean', 'paper']"]
        test["label"] = ["['paper', 'scratch']"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        with self.assertRaisesRegex(ValueError, "invalid list literal"):
            validate_audio_output(self.output_dir, "label", multi_label=True)

    def test_multi_label_double_quoted_strings_pass(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = ['["paper", "scratch"]', "['clean', 'paper']"]
        test["label"] = ["['paper', 'scratch']"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        validate_audio_output(self.output_dir, "label", multi_label=True)

    def test_multi_label_element_types_must_not_be_mixed(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = ["[1, 'scratch']", "[2, 3]"]
        test["label"] = ["[1, 2]"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        with self.assertRaisesRegex(ValueError, "only strings or only integers"):
            validate_audio_output(self.output_dir, "label", multi_label=True)

    def test_validation_without_scalar_target_passes(self):
        train = pd.read_csv(self.output_dir / "train.csv")[["file_name"]]
        test = pd.read_csv(self.output_dir / "test.csv")[["file_name"]]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        validate_audio_output(self.output_dir)

    def test_missing_train_manifest_fails(self):
        (self.output_dir / "train.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Required audio manifest is missing"):
            validate_audio_output(self.output_dir, "label")

    def test_missing_audio_directory_fails(self):
        (self.output_dir / "test" / "audio" / "tone_880.wav").unlink()
        (self.output_dir / "test" / "audio").rmdir()
        with self.assertRaisesRegex(ValueError, "Required audio directory is missing"):
            validate_audio_output(self.output_dir, "label")

    def test_empty_audio_directory_fails(self):
        (self.output_dir / "test" / "audio" / "tone_880.wav").unlink()
        with self.assertRaisesRegex(ValueError, "Required audio directory is empty"):
            validate_audio_output(self.output_dir, "label")

    def test_empty_manifest_fails(self):
        pd.DataFrame(columns=["file_name", "label"]).to_csv(
            self.output_dir / "train.csv", index=False
        )
        with self.assertRaisesRegex(ValueError, "at least one row"):
            validate_audio_output(self.output_dir, "label")

    def test_manifest_schema_mismatch_fails(self):
        test = pd.read_csv(self.output_dir / "test.csv")
        test["unexpected"] = 1
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "names or order"):
            validate_audio_output(self.output_dir, "label")

    def test_missing_file_name_column_fails(self):
        for split_name in ("train", "test"):
            manifest_path = self.output_dir / f"{split_name}.csv"
            manifest = pd.read_csv(manifest_path).drop(columns="file_name")
            manifest.to_csv(manifest_path, index=False)
        with self.assertRaisesRegex(ValueError, "missing required columns.*file_name"):
            validate_audio_output(self.output_dir, "label")

    def test_missing_file_name_value_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = None
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "missing file_name values"):
            validate_audio_output(self.output_dir, "label")

    def test_blank_file_name_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = "   "
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "empty or non-string file_name"):
            validate_audio_output(self.output_dir, "label")

    def test_non_string_file_name_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["file_name"] = [1, 2]
        test["file_name"] = [3]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "empty or non-string file_name"):
            validate_audio_output(self.output_dir, "label")

    def test_duplicate_file_name_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[1, "file_name"] = train.loc[0, "file_name"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "duplicate file_name values"):
            validate_audio_output(self.output_dir, "label")

    def test_shared_file_name_across_splits_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        test.loc[0, "file_name"] = train.loc[0, "file_name"]
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "same file_name values"):
            validate_audio_output(self.output_dir, "label")

    def test_missing_target_column_fails(self):
        for split_name in ("train", "test"):
            manifest_path = self.output_dir / f"{split_name}.csv"
            manifest = pd.read_csv(manifest_path).drop(columns="label")
            manifest.to_csv(manifest_path, index=False)
        with self.assertRaisesRegex(ValueError, "missing required columns.*label"):
            validate_audio_output(self.output_dir, "label")

    def test_constant_training_target_passes(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train["label"] = "scratch"
        train.to_csv(self.output_dir / "train.csv", index=False)
        validate_audio_output(self.output_dir, "label")

    def test_missing_audio_fails(self):
        self._write_audio("test/audio/unreferenced.wav", 1000)
        (self.output_dir / "test" / "audio" / "tone_880.wav").unlink()
        with self.assertRaisesRegex(ValueError, "does not exist"):
            validate_audio_output(self.output_dir, "label")

    def test_unreferenced_audio_file_fails(self):
        self._write_audio("train/audio/unreferenced.wav", 1000)
        with self.assertRaisesRegex(ValueError, "files not declared in manifest"):
            validate_audio_output(self.output_dir, "label")

    def test_byte_identical_audio_within_split_fail(self):
        source = self.output_dir / "train" / "audio" / "tone_440.wav"
        duplicate = self.output_dir / "train" / "audio" / "tone_440-copy.wav"
        duplicate.write_bytes(source.read_bytes())
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[len(train)] = ["train/audio/tone_440-copy.wav", "scratch"]
        train.to_csv(self.output_dir / "train.csv", index=False)

        with self.assertRaisesRegex(ValueError, "byte-identical audio files within train"):
            validate_audio_output(self.output_dir, "label")

    def test_path_outside_expected_split_fails(self):
        self._write_audio("train/audio/foreign.wav", 880)
        test = pd.read_csv(self.output_dir / "test.csv")
        test.loc[0, "file_name"] = "train/audio/foreign.wav"
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "must stay below test/audio/"):
            validate_audio_output(self.output_dir, "label")

    def test_path_outside_audio_directory_fails(self):
        self._write_audio("train/outside.wav", 1000)
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = "train/outside.wav"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "must stay below train/audio/"):
            validate_audio_output(self.output_dir, "label")

    def test_parent_path_traversal_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = "train/audio/../../../outside.wav"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            validate_audio_output(self.output_dir, "label")

    def test_absolute_path_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = str(
            (self.output_dir / "train" / "audio" / "tone_440.wav").resolve()
        )
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            validate_audio_output(self.output_dir, "label")

    def test_windows_path_separator_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = r"train\audio\tone_440.wav"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "portable POSIX separators"):
            validate_audio_output(self.output_dir, "label")

    def test_directory_instead_of_audio_file_fails(self):
        directory = self.output_dir / "train" / "audio" / "directory.wav"
        directory.mkdir()
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = "train/audio/directory.wav"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "is not a file"):
            validate_audio_output(self.output_dir, "label")

    def test_symlink_escaping_output_directory_fails(self):
        with tempfile.TemporaryDirectory() as external_directory:
            external_audio = Path(external_directory) / "external.wav"
            external_audio.write_bytes((self.output_dir / "train/audio/tone_440.wav").read_bytes())
            link = self.output_dir / "train" / "audio" / "external.wav"
            link.symlink_to(external_audio)
            train = pd.read_csv(self.output_dir / "train.csv")
            train.loc[0, "file_name"] = "train/audio/external.wav"
            train.to_csv(self.output_dir / "train.csv", index=False)
            with self.assertRaisesRegex(ValueError, "escapes output_dir"):
                validate_audio_output(self.output_dir, "label")

    def test_unseen_scalar_test_label_passes(self):
        test = pd.read_csv(self.output_dir / "test.csv")
        test.loc[0, "label"] = "unknown"
        test.to_csv(self.output_dir / "test.csv", index=False)
        validate_audio_output(self.output_dir, "label")

    def test_cross_split_duplicate_content_fails(self):
        source = self.output_dir / "train" / "audio" / "tone_440.wav"
        duplicate = self.output_dir / "test" / "audio" / "tone_880.wav"
        duplicate.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "byte-identical audio"):
            validate_audio_output(self.output_dir, "label")

    def test_cross_split_duplicate_check_can_be_disabled(self):
        source = self.output_dir / "train" / "audio" / "tone_440.wav"
        duplicate = self.output_dir / "test" / "audio" / "tone_880.wav"
        duplicate.write_bytes(source.read_bytes())

        validate_audio_output(
            self.output_dir,
            "label",
            check_content_overlap=False,
        )

    def test_within_split_duplicates_fail_with_cross_split_check_disabled(self):
        self._write_audio("train/audio/tone_660.wav", 440)

        with self.assertRaisesRegex(ValueError, "byte-identical audio files within train"):
            validate_audio_output(
                self.output_dir,
                "label",
                check_content_overlap=False,
            )




    def test_missing_test_manifest_fails(self):
        (self.output_dir / "test.csv").unlink()
        with self.assertRaisesRegex(ValueError, "manifest is missing"):
            validate_audio_output(self.output_dir, "label")

    def test_manifest_column_order_mismatch_fails(self):
        test = pd.read_csv(self.output_dir / "test.csv")[["label", "file_name"]]
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "names or order"):
            validate_audio_output(self.output_dir, "label")

    def test_manifest_dtype_mismatch_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["measurement"] = [1, 2]
        test["measurement"] = [1.5]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "dtypes differ"):
            validate_audio_output(self.output_dir, "label")

    def test_missing_or_blank_target_values_fail(self):
        # Two test rows keep the label dtype textual when one value is missing.
        test = pd.read_csv(self.output_dir / "test.csv")
        self._write_audio("test/audio/second.wav", 1000)
        test.loc[len(test)] = ["test/audio/second.wav", "clean"]
        test.to_csv(self.output_dir / "test.csv", index=False)
        for split in ("train", "test"):
            manifest_path = self.output_dir / f"{split}.csv"
            original = pd.read_csv(manifest_path)
            for value, message in [(None, "missing values"), ("   ", "blank labels")]:
                with self.subTest(split=split, value=value):
                    manifest = original.copy()
                    manifest.loc[0, "label"] = value
                    manifest.to_csv(manifest_path, index=False)
                    with self.assertRaisesRegex(ValueError, message):
                        validate_audio_output(self.output_dir, "label")
            original.to_csv(manifest_path, index=False)

    def test_normal_only_training_with_test_anomalies_passes(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = "normal"
        test["label"] = "normal"
        self._write_audio("test/audio/anomaly.wav", 1000)
        test.loc[len(test)] = ["test/audio/anomaly.wav", "anomaly"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)
        validate_audio_output(self.output_dir, "label")

    def test_signature_must_match_extension(self):
        path = self.output_dir / "test/audio/tone_880.wav"
        for content in (b"not audio", b"fLaC" + bytes(64)):
            with self.subTest(content=content[:8]):
                path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "signature does not match"):
                    validate_audio_output(self.output_dir, "label")

    def test_unsupported_extension_fails(self):
        path = self.output_dir / "test/audio/tone_880.wav"
        path.rename(path.with_suffix(".unknown"))
        test = pd.read_csv(self.output_dir / "test.csv")
        test.loc[0, "file_name"] = "test/audio/tone_880.unknown"
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Unsupported audio extension"):
            validate_audio_output(self.output_dir, "label")

    def test_damaged_wav_header_fails(self):
        # A WAV signature alone is insufficient: format and data chunks are missing.
        path = self.output_dir / "test/audio/tone_880.wav"
        path.write_bytes(b"RIFF" + (4).to_bytes(4, "little") + b"WAVE")
        with self.assertRaisesRegex(ValueError, "Unreadable audio"):
            validate_audio_output(self.output_dir, "label")

    def test_wav_without_samples_fails(self):
        path = self.output_dir / "test/audio/tone_880.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            stream.writeframes(b"")
        with self.assertRaisesRegex(ValueError, "no usable audio frames"):
            validate_audio_output(self.output_dir, "label")

    def test_truncated_wav_data_fails(self):
        # Preserve the header's frame count but remove the last audio samples.
        path = self.output_dir / "test/audio/tone_880.wav"
        path.write_bytes(path.read_bytes()[:-20])
        with self.assertRaisesRegex(ValueError, "incomplete audio frames"):
            validate_audio_output(self.output_dir, "label")


if __name__ == "__main__":
    unittest.main()
