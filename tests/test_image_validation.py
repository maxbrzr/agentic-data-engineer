import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from agentic_data_engineer.validation import (
    test_image_validation as validate_image_output,
)


class ImageValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary_directory.name)
        (self.output_dir / "train" / "images").mkdir(parents=True)
        (self.output_dir / "test" / "images").mkdir(parents=True)

        self._write_image("train/images/red.png", (255, 0, 0))
        self._write_image("train/images/green.png", (0, 255, 0))
        self._write_image("test/images/blue.png", (0, 0, 255))
        pd.DataFrame(
            {
                "file_name": [
                    "train/images/red.png",
                    "train/images/green.png",
                ],
                "label": ["scratch", "clean"],
            }
        ).to_csv(self.output_dir / "train.csv", index=False)
        pd.DataFrame(
            {
                "file_name": ["test/images/blue.png"],
                "label": ["scratch"],
            }
        ).to_csv(self.output_dir / "test.csv", index=False)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_image(self, relative_path: str, color: tuple[int, int, int]) -> None:
        Image.new("RGB", (4, 4), color).save(self.output_dir / relative_path)

    def test_valid_image_splits_pass(self):
        validate_image_output(self.output_dir, "label")

    def test_valid_multi_label_splits_pass_with_one_row_per_image(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = [
            repr(["paper", "scratch"]),
            repr(["clean", "paper"]),
        ]
        test["label"] = [repr(["paper", "scratch"])]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        validate_image_output(self.output_dir, "label", multi_label=True)
        self.assertEqual(len(pd.read_csv(self.output_dir / "train.csv")), 2)
        self.assertEqual(len(pd.read_csv(self.output_dir / "test.csv")), 1)

    def test_valid_integer_multi_label_splits_pass(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = [repr([1, 2]), repr([2, 3])]
        test["label"] = [repr([1, 3])]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        validate_image_output(self.output_dir, "label", multi_label=True)

    def test_inline_captions_and_objects_pass_without_sidecars(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["captions"] = [repr(["red sample", "warm color"]), repr(["green sample"])]
        test["captions"] = [repr(["blue sample"])]
        train["objects"] = [
            repr([{"label": "mark", "xmin": 0, "ymin": 0, "xmax": 2, "ymax": 2}]),
            repr([]),
        ]
        test["objects"] = [repr([])]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        validate_image_output(
            self.output_dir,
            "label",
            list_columns=["captions", "objects"],
        )
        self.assertFalse((self.output_dir / "train_captions.csv").exists())
        self.assertFalse((self.output_dir / "train_objects.csv").exists())

    def test_inline_annotation_column_must_use_list_literal(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["captions"] = ["red sample", repr(["green sample"])]
        test["captions"] = [repr(["blue sample"])]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        with self.assertRaisesRegex(ValueError, "invalid list literal"):
            validate_image_output(
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
            validate_image_output(self.output_dir, "label", multi_label=True)

    def test_multi_label_lists_must_be_sorted_and_unique(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = ["['scratch', 'paper']", "['clean', 'paper']"]
        test["label"] = ["['paper', 'scratch']"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        with self.assertRaisesRegex(ValueError, "sort labels deterministically"):
            validate_image_output(self.output_dir, "label", multi_label=True)

        train.loc[0, "label"] = "['paper', 'paper']"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "duplicate labels"):
            validate_image_output(self.output_dir, "label", multi_label=True)

    def test_multi_label_cell_must_be_a_list_literal(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = ["paper,scratch", "['clean', 'paper']"]
        test["label"] = ["['paper', 'scratch']"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        with self.assertRaisesRegex(ValueError, "invalid list literal"):
            validate_image_output(self.output_dir, "label", multi_label=True)

    def test_json_style_string_array_is_not_canonical(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = ['["paper", "scratch"]', "['clean', 'paper']"]
        test["label"] = ["['paper', 'scratch']"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        with self.assertRaisesRegex(ValueError, "canonical Python list"):
            validate_image_output(self.output_dir, "label", multi_label=True)

    def test_multi_label_element_types_must_not_be_mixed(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["label"] = ["[1, 'scratch']", "[2, 3]"]
        test["label"] = ["[1, 2]"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        with self.assertRaisesRegex(ValueError, "only strings or only integers"):
            validate_image_output(self.output_dir, "label", multi_label=True)

    def test_validation_without_scalar_target_passes(self):
        train = pd.read_csv(self.output_dir / "train.csv")[["file_name"]]
        test = pd.read_csv(self.output_dir / "test.csv")[["file_name"]]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)

        validate_image_output(self.output_dir)

    def test_missing_train_manifest_fails(self):
        (self.output_dir / "train.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Required image manifest is missing"):
            validate_image_output(self.output_dir, "label")

    def test_missing_image_directory_fails(self):
        (self.output_dir / "test" / "images" / "blue.png").unlink()
        (self.output_dir / "test" / "images").rmdir()
        with self.assertRaisesRegex(ValueError, "Required image directory is missing"):
            validate_image_output(self.output_dir, "label")

    def test_empty_image_directory_fails(self):
        (self.output_dir / "test" / "images" / "blue.png").unlink()
        with self.assertRaisesRegex(ValueError, "Required image directory is empty"):
            validate_image_output(self.output_dir, "label")

    def test_empty_manifest_fails(self):
        pd.DataFrame(columns=["file_name", "label"]).to_csv(
            self.output_dir / "train.csv", index=False
        )
        with self.assertRaisesRegex(ValueError, "at least one row"):
            validate_image_output(self.output_dir, "label")

    def test_manifest_schema_mismatch_fails(self):
        test = pd.read_csv(self.output_dir / "test.csv")
        test["unexpected"] = 1
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "names or order"):
            validate_image_output(self.output_dir, "label")

    def test_missing_file_name_column_fails(self):
        for split_name in ("train", "test"):
            manifest_path = self.output_dir / f"{split_name}.csv"
            manifest = pd.read_csv(manifest_path).drop(columns="file_name")
            manifest.to_csv(manifest_path, index=False)
        with self.assertRaisesRegex(ValueError, "missing required columns.*file_name"):
            validate_image_output(self.output_dir, "label")

    def test_missing_file_name_value_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = None
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "missing file_name values"):
            validate_image_output(self.output_dir, "label")

    def test_blank_file_name_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = "   "
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "empty or non-string file_name"):
            validate_image_output(self.output_dir, "label")

    def test_non_string_file_name_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        train["file_name"] = [1, 2]
        test["file_name"] = [3]
        train.to_csv(self.output_dir / "train.csv", index=False)
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "empty or non-string file_name"):
            validate_image_output(self.output_dir, "label")

    def test_duplicate_file_name_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[1, "file_name"] = train.loc[0, "file_name"]
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "duplicate file_name values"):
            validate_image_output(self.output_dir, "label")

    def test_shared_file_name_across_splits_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        test = pd.read_csv(self.output_dir / "test.csv")
        test.loc[0, "file_name"] = train.loc[0, "file_name"]
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "same file_name values"):
            validate_image_output(self.output_dir, "label")

    def test_missing_target_column_fails(self):
        for split_name in ("train", "test"):
            manifest_path = self.output_dir / f"{split_name}.csv"
            manifest = pd.read_csv(manifest_path).drop(columns="label")
            manifest.to_csv(manifest_path, index=False)
        with self.assertRaisesRegex(ValueError, "missing required columns.*label"):
            validate_image_output(self.output_dir, "label")

    def test_constant_training_target_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train["label"] = "scratch"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "constant in Training split"):
            validate_image_output(self.output_dir, "label")

    def test_missing_image_fails(self):
        self._write_image("test/images/unreferenced.png", (255, 255, 0))
        (self.output_dir / "test" / "images" / "blue.png").unlink()
        with self.assertRaisesRegex(ValueError, "does not exist"):
            validate_image_output(self.output_dir, "label")

    def test_unreadable_image_fails(self):
        (self.output_dir / "test" / "images" / "blue.png").write_text(
            "not an image", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Unreadable image"):
            validate_image_output(self.output_dir, "label")

    def test_unreferenced_image_file_fails(self):
        self._write_image("train/images/unreferenced.png", (255, 255, 0))
        with self.assertRaisesRegex(ValueError, "files not declared in manifest"):
            validate_image_output(self.output_dir, "label")

    def test_byte_identical_images_within_split_fail(self):
        source = self.output_dir / "train" / "images" / "red.png"
        duplicate = self.output_dir / "train" / "images" / "red-copy.png"
        duplicate.write_bytes(source.read_bytes())
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[len(train)] = ["train/images/red-copy.png", "scratch"]
        train.to_csv(self.output_dir / "train.csv", index=False)

        with self.assertRaisesRegex(ValueError, "byte-identical images within train"):
            validate_image_output(self.output_dir, "label")

    def test_pixel_identical_images_within_split_fail(self):
        source = self.output_dir / "train" / "images" / "red.png"
        duplicate = self.output_dir / "train" / "images" / "red-reencoded.png"
        with Image.open(source) as image:
            image.save(duplicate, format="PNG", compress_level=0)
        self.assertNotEqual(source.read_bytes(), duplicate.read_bytes())

        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[len(train)] = ["train/images/red-reencoded.png", "scratch"]
        train.to_csv(self.output_dir / "train.csv", index=False)

        with self.assertRaisesRegex(ValueError, "pixel-identical images within train"):
            validate_image_output(self.output_dir, "label")

    def test_path_outside_expected_split_fails(self):
        self._write_image("train/images/foreign.png", (0, 0, 255))
        test = pd.read_csv(self.output_dir / "test.csv")
        test.loc[0, "file_name"] = "train/images/foreign.png"
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "must stay below test/images/"):
            validate_image_output(self.output_dir, "label")

    def test_path_outside_images_directory_fails(self):
        self._write_image("train/outside.png", (255, 255, 0))
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = "train/outside.png"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "must stay below train/images/"):
            validate_image_output(self.output_dir, "label")

    def test_parent_path_traversal_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = "train/images/../../../outside.png"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            validate_image_output(self.output_dir, "label")

    def test_absolute_path_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = str(
            (self.output_dir / "train" / "images" / "red.png").resolve()
        )
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            validate_image_output(self.output_dir, "label")

    def test_windows_path_separator_fails(self):
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = r"train\images\red.png"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "portable POSIX separators"):
            validate_image_output(self.output_dir, "label")

    def test_directory_instead_of_image_file_fails(self):
        directory = self.output_dir / "train" / "images" / "directory.png"
        directory.mkdir()
        train = pd.read_csv(self.output_dir / "train.csv")
        train.loc[0, "file_name"] = "train/images/directory.png"
        train.to_csv(self.output_dir / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "is not a file"):
            validate_image_output(self.output_dir, "label")

    def test_symlink_escaping_output_directory_fails(self):
        with tempfile.TemporaryDirectory() as external_directory:
            external_image = Path(external_directory) / "external.png"
            Image.new("RGB", (4, 4), (255, 255, 0)).save(external_image)
            link = self.output_dir / "train" / "images" / "external.png"
            link.symlink_to(external_image)
            train = pd.read_csv(self.output_dir / "train.csv")
            train.loc[0, "file_name"] = "train/images/external.png"
            train.to_csv(self.output_dir / "train.csv", index=False)
            with self.assertRaisesRegex(ValueError, "escapes output_dir"):
                validate_image_output(self.output_dir, "label")

    def test_unseen_test_label_fails(self):
        test = pd.read_csv(self.output_dir / "test.csv")
        test.loc[0, "label"] = "unknown"
        test.to_csv(self.output_dir / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "not present in training"):
            validate_image_output(self.output_dir, "label")

    def test_cross_split_duplicate_content_fails(self):
        source = self.output_dir / "train" / "images" / "red.png"
        duplicate = self.output_dir / "test" / "images" / "blue.png"
        duplicate.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "byte-identical image"):
            validate_image_output(self.output_dir, "label")

    def test_cross_split_duplicate_check_can_be_disabled(self):
        source = self.output_dir / "train" / "images" / "red.png"
        duplicate = self.output_dir / "test" / "images" / "blue.png"
        duplicate.write_bytes(source.read_bytes())

        validate_image_output(
            self.output_dir,
            "label",
            check_content_overlap=False,
        )


if __name__ == "__main__":
    unittest.main()
