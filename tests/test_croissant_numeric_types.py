import json
import tempfile
import unittest
from pathlib import Path

from agentic_data_engineer.validation import test_croissant_numeric_columns_have_numeric_types


class NumericTypeTests(unittest.TestCase):
    def check_column(self, values, data_type):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'test'):
                (root / f'{split}.csv').write_text('d1v\n' + '\n'.join(values) + '\n')
            metadata = {'recordSet': [
                {'name': split, 'field': [{'name': 'd1v', 'dataType': data_type}]}
                for split in ('train', 'test')
            ]}
            path = root / 'croissant.json'
            path.write_text(json.dumps(metadata))
            test_croissant_numeric_columns_have_numeric_types(path)

    def test_late_text_after_numeric_chunk_is_valid(self):
        self.check_column(['22'] * 10001 + ['A'], 'sc:Text')

    def test_fully_numeric_text_field_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Numeric CSV columns'):
            self.check_column(['22'] * 10001 + ['18'], 'sc:Text')

    def test_numeric_field_passes(self):
        self.check_column(['22', '18'], 'sc:Integer')
