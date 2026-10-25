"""Run a dataset folder's processor the way dcat-ap-hub's Dataset.process() does.

Usage: run_processor.py <dataset folder>

Imports <dataset folder>/processor.py and calls its module-level
``process(data_dir, output_dir)`` with output_dir = <dataset folder>/processed.
"""

import importlib.util
import inspect
import sys
from pathlib import Path


def main(data_dir: Path) -> int:
    script = data_dir / "processor.py"
    output_dir = data_dir / "processed"
    output_dir.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("processor", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    process = getattr(module, "process", None)
    if not inspect.isfunction(process):
        print(f"{script} defines no module-level process(data_dir, output_dir)", file=sys.stderr)
        return 2
    process(data_dir, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
