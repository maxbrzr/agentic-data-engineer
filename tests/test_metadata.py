import json
import tempfile
import unittest
from pathlib import Path

from agentic_data_engineer.contracts import (
    DatasetSpec,
    RetrievedDataset,
)
from agentic_data_engineer.metadata import CroissantBakerMetadataGenerator


class CroissantBakerMetadataGeneratorTests(unittest.TestCase):
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
