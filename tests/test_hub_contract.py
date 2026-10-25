"""Contract between this pipeline and dcat-ap-hub, exercised over real HTTP.

Covers both ends of the hub in the VISION.md lifecycle:

1. raw record -> hub download -> (fake) agent output -> dataset package ->
   ``Dataset.from_directory().process()`` reproduces train/test;
2. the package published under a URL -> ``Dataset.from_url()`` -> download ->
   ``process()`` reproduces train/test.

The agent itself is replaced by a fixed parser.py, so this test checks only the
interfaces, not the model.
"""

import functools
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dcat_ap_hub import Dataset
from dcat_ap_hub.metadata.constants import PROCESSOR_PROFILE_URIS
from dcat_ap_hub.processing.processor import apply_processor_logic

from agentic_data_engineer.pipeline import PROCESSOR_PROFILE_URI, DataEngineeringPipeline
from agentic_data_engineer.retrieval import DcatApHubRetriever, dataset_spec_from_url

# Stands in for the agent's generated processor; follows the prompt's contract.
PARSER = '''
from pathlib import Path

import pandas as pd


def parse(input_files):
    tables = sorted(p for p in input_files if p.suffix == ".csv")
    frames = [pd.read_csv(p, float_precision="round_trip") for p in tables]
    return pd.concat(frames, ignore_index=True).sort_values("id", kind="stable")


def process(input_files, output_dir):
    df = parse([Path(p) for p in input_files])
    test = df["id"] % 5 == 0
    df[~test].to_csv(Path(output_dir) / "train.csv", index=False)
    df[test].to_csv(Path(output_dir) / "test.csv", index=False)
'''


class _Handler(SimpleHTTPRequestHandler):
    # Serve scripts as text/plain, as raw.githubusercontent.com does.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".py": "text/plain",
        ".jsonld": "application/ld+json",
    }

    def log_message(self, *args):
        pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HubContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.remote = self.root / "remote"
        self.remote.mkdir()
        handler = functools.partial(_Handler, directory=str(self.remote))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        self.record_url = self._publish_raw_record()

    def _publish_raw_record(self) -> str:
        """A piveau-shaped raw record whose files live at the 'source portal'."""
        portal = self.remote / "portal"
        portal.mkdir()
        rows = [f"{i},{i * 0.1:.17g},{i % 2}" for i in range(1, 41)]
        (portal / "part1.csv").write_text("id,value,label\n" + "\n".join(rows[:20]) + "\n")
        (portal / "part2.csv").write_text("id,value,label\n" + "\n".join(rows[20:]) + "\n")
        dataset_id = "https://piveau.io/set/data/pump-sensor-data"
        record = {
            "@graph": [
                {
                    "@id": dataset_id,
                    "@type": "dcat:Dataset",
                    # ':' and '/' must not leak into folder names (I-17).
                    "dct:title": {"@language": "en", "@value": "Pump: Sensor/Data"},
                    "dct:description": {"@language": "en", "@value": "Fixture."},
                    "dcat:distribution": {"@id": f"{dataset_id}#files"},
                },
                {
                    "@id": f"{dataset_id}#files",
                    "@type": "dcat:Distribution",
                    "dct:title": {"@language": "en", "@value": "part.csv"},
                    # Several download URLs on one distribution (I-11).
                    "dcat:downloadURL": [
                        {"@id": f"{self.base}/portal/part1.csv"},
                        {"@id": f"{self.base}/portal/part2.csv"},
                    ],
                },
            ]
        }
        (self.remote / "raw.jsonld").write_text(json.dumps(record))
        return f"{self.base}/raw.jsonld"

    def _run_fake_agent(self):
        """Retrieve through the hub, then do what a successful agent run leaves behind."""
        spec = dataset_spec_from_url(self.record_url)
        retrieved = DcatApHubRetriever(verbose=False).retrieve(spec, self.root / "data")
        output_dir = self.root / "output" / spec.key / "run"
        output_dir.mkdir(parents=True)
        (output_dir / "parser.py").write_text(PARSER)
        inputs = [
            path for path in retrieved.files if path.name != "dcat-metadata.jsonld"
        ]
        apply_processor_logic(output_dir / "parser.py", inputs, output_dir)
        DataEngineeringPipeline._require_agent_outputs(output_dir, "fake-run")
        DataEngineeringPipeline.copy_source_data(retrieved.data_dir, output_dir)
        DataEngineeringPipeline.build_dataset_package(output_dir, spec)
        return retrieved, output_dir

    def assertReproduces(self, processed: Path, output_dir: Path):
        for name in ("train.csv", "test.csv"):
            self.assertEqual(
                _sha256(processed / name), _sha256(output_dir / name), name
            )

    def test_processor_profile_matches_hub(self):
        self.assertIn(PROCESSOR_PROFILE_URI, PROCESSOR_PROFILE_URIS)

    def test_hub_folder_is_safe_and_holds_every_file(self):
        retrieved, _ = self._run_fake_agent()
        self.assertEqual(retrieved.data_dir.name, "Pump Sensor Data")
        self.assertEqual(retrieved.metadata_title, "Pump: Sensor/Data")
        csvs = [path for path in retrieved.files if path.suffix == ".csv"]
        self.assertEqual(len(csvs), 2)
        self.assertTrue((retrieved.data_dir / "dcat-metadata.jsonld").is_file())

    def test_run_package_processes_locally(self):
        _, output_dir = self._run_fake_agent()

        ds = Dataset.from_directory(output_dir / "data")
        ds.process(verbose=False)

        self.assertReproduces(ds.processed_path, output_dir)

    def test_published_package_processes_from_url(self):
        _, output_dir = self._run_fake_agent()
        # Stand-in for the missing publish step (VISION.md I-8): host parser.py
        # and turn the draft's relative processor URL into an absolute one.
        hosted = self.remote / "ai-ready"
        hosted.mkdir()
        (hosted / "parser.py").write_bytes((output_dir / "data" / "parser.py").read_bytes())
        draft = json.loads((output_dir / "data" / "dcat-metadata.jsonld").read_text())
        for node in draft["@graph"]:
            if node.get("dcat:downloadURL") == "parser.py":
                node["dcat:downloadURL"] = {"@id": f"{self.base}/ai-ready/parser.py"}
        (hosted / "record.jsonld").write_text(json.dumps(draft))

        ds = Dataset.from_url(f"{self.base}/ai-ready/record.jsonld")
        ds.download(data_dir=self.root / "consumer", verbose=False)
        ds.process(verbose=False)

        self.assertReproduces(ds.processed_path, output_dir)


if __name__ == "__main__":
    unittest.main()
