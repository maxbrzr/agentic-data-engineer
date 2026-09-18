import base64
import json
import tempfile
import unittest
import wave
from pathlib import Path

from agentic_data_engineer.contracts import (
    DatasetSpec,
    RetrievedDataset,
)
from agentic_data_engineer.metadata import CroissantBakerMetadataGenerator
from agentic_data_engineer.metadata.audio_handler import AudioHandler


class CroissantBakerMetadataGeneratorTests(unittest.TestCase):
    def test_curated_mimii_dg_metadata_passes_preflight(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "agentic_data_engineer"
            / "retrieval"
            / "metadata"
            / "mimii-dg.jsonld"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir).resolve()
            (data_dir / "dcat-metadata.jsonld").write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            spec = DatasetSpec(
                "mimii-dg",
                "MIMII DG",
                "https://zenodo.org/records/6355122",
            )

            CroissantBakerMetadataGenerator().preflight(
                RetrievedDataset(spec, data_dir, spec.title)
            )

    def test_audio_handler_rejects_extension_without_audio_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_audio = Path(temp_dir) / "fake.wav"
            fake_audio.write_text("not audio", encoding="utf-8")

            self.assertFalse(AudioHandler().can_handle(fake_audio))

    def test_generates_valid_train_test_metadata_from_dcat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            output_dir.mkdir()

            metadata = {
                "@graph": [
                    {
                        "@id": "https://example.test/creator",
                        "@type": "foaf:Person",
                        "foaf:name": "Test Creator",
                    },
                    {
                        "@id": "https://example.test/distribution",
                        "@type": "dcat:Distribution",
                        "dct:license": "cc-by-4.0",
                    },
                    {
                        "@id": "https://example.test/dataset",
                        "@type": "dcat:Dataset",
                        "dct:title": {"@language": "en", "@value": "Test Dataset"},
                        "dct:description": {
                            "@language": "en",
                            "@value": "A test dataset with prepared ML splits.",
                        },
                        "dct:creator": [{"@id": "https://example.test/creator"}],
                        "dct:issued": {"@value": "2026-01-02"},
                        "dct:modified": {"@value": "2026-01-03"},
                        "dct:identifier": "10.1234/example",
                        "dcat:keyword": [{"@value": "testing"}],
                    },
                ]
            }
            (data_dir / "dcat-metadata.jsonld").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            (output_dir / "train.csv").write_text(
                "feature,target\n1.5,0\n2.5,1\n",
                encoding="utf-8",
            )
            (output_dir / "test.csv").write_text(
                "feature,target\n3.5,0\n4.5,1\n",
                encoding="utf-8",
            )
            spec = DatasetSpec(
                key="test",
                title="Fallback title",
                url="https://example.test/catalogue",
            )
            dataset = RetrievedDataset(
                spec=spec,
                data_dir=data_dir,
                metadata_title=spec.title,
            )

            result = CroissantBakerMetadataGenerator().generate(
                dataset,
                output_dir,
            )

            self.assertEqual("croissant-baker", result.generator)
            self.assertEqual(output_dir / "croissant.json", result.path)
            generated = json.loads(result.path.read_text(encoding="utf-8"))
            self.assertEqual("Test Dataset", generated["name"])
            self.assertEqual(
                "https://creativecommons.org/licenses/by/4.0/",
                generated["license"],
            )
            self.assertEqual(
                {"train.csv", "test.csv"},
                {
                    distribution["name"]
                    for distribution in generated["distribution"]
                },
            )
            self.assertEqual(
                {"train", "test"},
                {record_set["name"] for record_set in generated["recordSet"]},
            )

    def test_missing_split_fails_before_baker_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            output_dir.mkdir()
            spec = DatasetSpec("test", "Test", "https://example.test")
            dataset = RetrievedDataset(spec, data_dir, spec.title)

            with self.assertRaisesRegex(ValueError, "missing agent outputs"):
                CroissantBakerMetadataGenerator().generate(dataset, output_dir)

    def test_generates_image_resources_alongside_split_manifests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            output_dir.mkdir()

            metadata = {
                "@graph": [
                    {
                        "@id": "https://example.test/creator",
                        "@type": "foaf:Person",
                        "foaf:name": "Image Creator",
                    },
                    {
                        "@type": "dcat:Distribution",
                        "dct:license": "cc-by-4.0",
                    },
                    {
                        "@type": "dcat:Dataset",
                        "dct:title": "Prepared Image Dataset",
                        "dct:description": "Images with classification manifests.",
                        "dct:creator": {"@id": "https://example.test/creator"},
                        "dct:issued": {"@value": "2026-08-14"},
                        "dct:identifier": "10.1234/images",
                    },
                ]
            }
            (data_dir / "dcat-metadata.jsonld").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                "+A8AAQUBAScY42YAAAAASUVORK5CYII="
            )
            for split, target in (("train", 0), ("test", 1)):
                image_dir = output_dir / split / "images"
                mask_dir = output_dir / split / "masks"
                image_dir.mkdir(parents=True)
                mask_dir.mkdir(parents=True)
                image_name = f"{split}.png"
                (image_dir / image_name).write_bytes(png)
                (mask_dir / image_name).write_bytes(png)
                (output_dir / f"{split}.csv").write_text(
                    "file_name,mask_file_name,label\n"
                    f"{split}/images/{image_name},{split}/masks/{image_name},"
                    f"class-{target}\n",
                    encoding="utf-8",
                )
            (output_dir / "mask_labels.csv").write_text(
                "mask_value,label\n0,background\n1,object\n",
                encoding="utf-8",
            )

            spec = DatasetSpec(
                "images",
                "Image Dataset",
                "https://example.test/images",
            )
            dataset = RetrievedDataset(spec, data_dir, spec.title)

            result = CroissantBakerMetadataGenerator().generate(dataset, output_dir)
            generated = json.loads(result.path.read_text(encoding="utf-8"))

            self.assertIn(
                "cr:FileSet",
                {item["@type"] for item in generated["distribution"]},
            )
            self.assertTrue(
                {
                    "train/images/train.png",
                    "test/images/test.png",
                    "train/masks/train.png",
                    "test/masks/test.png",
                }
                <= {
                    item.get("contentUrl")
                    for item in generated["distribution"]
                    if item.get("@type") == "cr:FileObject"
                }
            )
            image_record_set = next(
                item for item in generated["recordSet"] if item["name"] == "images"
            )
            self.assertIn(
                "sc:ImageObject",
                {
                    field.get("dataType")
                    for field in image_record_set["field"]
                },
            )
            self.assertIn(
                "mask_labels",
                {item["name"] for item in generated["recordSet"]},
            )

    def test_generates_audio_resources_alongside_split_manifests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            output_dir.mkdir()

            metadata = {
                "@graph": [
                    {
                        "@id": "https://example.test/creator",
                        "@type": "foaf:Person",
                        "foaf:name": "Audio Creator",
                    },
                    {
                        "@type": "dcat:Distribution",
                        "dct:license": "cc-by-4.0",
                    },
                    {
                        "@type": "dcat:Dataset",
                        "dct:title": "Prepared Audio Dataset",
                        "dct:description": "Audio with classification manifests.",
                        "dct:creator": {"@id": "https://example.test/creator"},
                        "dct:issued": {"@value": "2026-08-14"},
                        "dct:identifier": "10.1234/audio",
                    },
                ]
            }
            (data_dir / "dcat-metadata.jsonld").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            for split, label in (("train", "normal"), ("test", "fault")):
                audio_dir = output_dir / split / "audio"
                audio_dir.mkdir(parents=True)
                audio_name = f"{split}.wav"
                with wave.open(str(audio_dir / audio_name), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(8000)
                    wav.writeframes(b"\x00\x00" * 80)
                (output_dir / f"{split}.csv").write_text(
                    "file_name,label\n"
                    f"{split}/audio/{audio_name},{label}\n",
                    encoding="utf-8",
                )
                (output_dir / f"{split}_captions.csv").write_text(
                    "file_name,text\n"
                    f"{split}/audio/{audio_name},A short audio caption.\n",
                    encoding="utf-8",
                )

            spec = DatasetSpec(
                "audio",
                "Audio Dataset",
                "https://example.test/audio",
            )
            dataset = RetrievedDataset(spec, data_dir, spec.title)

            result = CroissantBakerMetadataGenerator().generate(dataset, output_dir)
            generated = json.loads(result.path.read_text(encoding="utf-8"))

            self.assertTrue(
                {"train/audio/train.wav", "test/audio/test.wav"}
                <= {
                    item.get("contentUrl")
                    for item in generated["distribution"]
                    if item.get("@type") == "cr:FileObject"
                }
            )
            audio_record_set = next(
                item for item in generated["recordSet"] if item["name"] == "audio"
            )
            self.assertIn(
                "sc:AudioObject",
                {
                    field.get("dataType")
                    for field in audio_record_set["field"]
                },
            )
            self.assertTrue(
                {"train_captions", "test_captions"}
                <= {item["name"] for item in generated["recordSet"]}
            )

    def test_unresolved_zenodo_creator_uri_supplies_creator_name(self):
        captured = {}

        class FakeBaker:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def save_metadata(self, path, *, validate):
                Path(path).write_text("{}\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            output_dir.mkdir()
            metadata = {
                "@graph": [
                    {
                        "@type": "dcat:Distribution",
                        "dct:license": "cc-by-4.0",
                    },
                    {
                        "@type": "dcat:Dataset",
                        "dct:title": "Chemical Safety",
                        "dct:description": "Prepared chemical safety records.",
                        "dct:creator": {
                            "@id": (
                                "https://zenodo.org/author/"
                                "Acuña_Acuña,_Edwin_Gerardo"
                            )
                        },
                        "dct:issued": {"@value": "2026-01-09"},
                        "dct:identifier": "10.5281/zenodo.18200704",
                    },
                ]
            }
            (data_dir / "dcat-metadata.jsonld").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            for name in ("train.csv", "test.csv"):
                (output_dir / name).write_text(
                    "feature,target\n1,0\n",
                    encoding="utf-8",
                )
            spec = DatasetSpec(
                "chemical-process-safety",
                "Chemical Safety",
                "https://example.test/chemical",
            )
            dataset = RetrievedDataset(spec, data_dir, spec.title)
            generator = CroissantBakerMetadataGenerator(
                generator_factory=FakeBaker,
                validator=lambda _path: None,
            )

            generator.preflight(dataset)
            generator.generate(dataset, output_dir)

            self.assertEqual(
                [{"name": "Acuña Acuña, Edwin Gerardo"}],
                captured["creators"],
            )

    def test_darus_metadata_variants_support_preflight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            data_dir.mkdir()
            metadata = {
                "@graph": [
                    {
                        "@type": "dcat:Distribution",
                        "dct:license": {
                            "@id": "https://creativecommons.org/licenses/by/4.0/"
                        },
                    },
                    {
                        "@type": "dcat:Dataset",
                        "dct:title": "Floor Type Detection Dataset",
                        "dct:description": "Prepared multimodal robot observations.",
                        "dct:creator": {
                            "@id": "https://darus.uni-stuttgart.de/author/Eißen,_Dominik"
                        },
                        "dcat:issued": {"@value": "2024-08-12T13:31:45Z"},
                        "dcat:landingPage": {
                            "@id": (
                                "https://darus.uni-stuttgart.de/dataset.xhtml?"
                                "persistentId=doi:10.18419/DARUS-4353"
                            )
                        },
                    },
                ]
            }
            (data_dir / "dcat-metadata.jsonld").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            spec = DatasetSpec(
                "floor-type-detection",
                "Floor Type Detection Dataset",
                "https://example.test/floor",
            )

            CroissantBakerMetadataGenerator().preflight(
                RetrievedDataset(spec, data_dir, spec.title)
            )

    def test_preflight_rejects_missing_source_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "dcat-metadata.jsonld").write_text(
                json.dumps(
                    {
                        "@graph": [
                            {
                                "@type": "dcat:Dataset",
                                "dct:title": "Incomplete Dataset",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            spec = DatasetSpec(
                "incomplete",
                "Incomplete Dataset",
                "https://example.test/incomplete",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"description.*license.*citation.*date_published.*creators",
            ):
                CroissantBakerMetadataGenerator().preflight(
                    RetrievedDataset(spec, data_dir, spec.title)
                )


if __name__ == "__main__":
    unittest.main()
