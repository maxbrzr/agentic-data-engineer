import unittest

import numpy as np
import pandas as pd

from agentic_data_engineer.validation import (
    test_chronological_split,
    test_classification_labels,
    test_datetimeindex_consistency,
    test_deterministic_splits,
    test_no_group_leakage,
    test_no_infinite_values,
    test_no_shared_rows,
    test_split_schema,
    test_tabular_splits,
    test_timestamps,
    test_unique_columns,
)


def classification_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "row_id": pd.Series([1, 2, 3, 4], dtype="int64"),
            "group_id": pd.Series([10, 10, 11, 11], dtype="int64"),
            "feature": pd.Series([0.1, 0.2, 0.3, 0.4], dtype="float64"),
            "target": pd.Series([0, 1, 0, 1], dtype="int64"),
        }
    )
    test = pd.DataFrame(
        {
            "row_id": pd.Series([5, 6], dtype="int64"),
            "group_id": pd.Series([12, 12], dtype="int64"),
            "feature": pd.Series([0.5, 0.6], dtype="float64"),
            "target": pd.Series([0, 1], dtype="int64"),
        }
    )
    return train, test


class DataFrameValidationTests(unittest.TestCase):
    def test_generic_classification_splits_pass(self):
        train, test = classification_splits()
        test_tabular_splits(
            train,
            test,
            "target",
            task_type="classification",
            key_columns=["row_id"],
            group_columns=["group_id"],
        )

    def test_duplicate_columns_fail(self):
        frame = pd.DataFrame([[1, 2]], columns=["feature", "feature"])
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            test_unique_columns(frame)

    def test_infinite_values_fail(self):
        frame = pd.DataFrame({"feature": [1.0, np.inf]})
        with self.assertRaisesRegex(ValueError, "Infinite values"):
            test_no_infinite_values(frame)

    def test_target_cannot_be_excluded_from_numeric_validation(self):
        train, test = classification_splits()
        with self.assertRaisesRegex(ValueError, "target column cannot be excluded"):
            test_tabular_splits(
                train,
                test,
                "target",
                task_type="classification",
                exclude_from_numeric=["target"],
                key_columns=["row_id"],
            )

    def test_schema_order_mismatch_fails(self):
        train, test = classification_splits()
        test = test[["group_id", "row_id", "feature", "target"]]
        with self.assertRaisesRegex(ValueError, "names or order"):
            test_split_schema(train, test)

    def test_shared_observation_key_fails(self):
        train, test = classification_splits()
        test.loc[0, "row_id"] = train.loc[0, "row_id"]
        with self.assertRaisesRegex(ValueError, "shared observation"):
            test_no_shared_rows(train, test, ["row_id"])

    def test_group_leakage_fails(self):
        train, test = classification_splits()
        test.loc[0, "group_id"] = train.loc[0, "group_id"]
        with self.assertRaisesRegex(ValueError, "protected group"):
            test_no_group_leakage(train, test, ["group_id"])

    def test_unseen_classification_label_fails(self):
        train, test = classification_splits()
        test.loc[0, "target"] = 2
        with self.assertRaisesRegex(ValueError, "not present in training"):
            test_classification_labels(train, test, "target")

    def test_non_contiguous_classification_labels_fail(self):
        train, test = classification_splits()
        train["target"] = train["target"].replace({1: 2})
        test["target"] = test["target"].replace({1: 2})
        with self.assertRaisesRegex(ValueError, "contiguous integers"):
            test_classification_labels(train, test, "target")

    def test_determinism_checks_row_order(self):
        train, test = classification_splits()
        changed_train = train.iloc[::-1]
        with self.assertRaisesRegex(ValueError, "different splits"):
            test_deterministic_splits(train, test, changed_train, test.copy())


class TemporalValidationTests(unittest.TestCase):
    def test_regular_datetime_index_passes(self):
        frame = pd.DataFrame(
            {"feature": [1.0, 2.0, 3.0]},
            index=pd.date_range("2026-01-01", periods=3, freq="15min"),
        )
        test_datetimeindex_consistency(frame)
        test_timestamps(frame, "15min")

    def test_irregular_datetime_index_fails(self):
        frame = pd.DataFrame(
            {"feature": [1.0, 2.0, 3.0]},
            index=pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:45"]
            ),
        )
        with self.assertRaisesRegex(ValueError, "frequency mismatch"):
            test_timestamps(frame, "15min")

    def test_chronological_split_passes(self):
        train = pd.DataFrame(
            {"time": ["2026-01-01", "2026-01-02"], "target": [0, 1]}
        )
        test = pd.DataFrame(
            {"time": ["2026-01-03", "2026-01-04"], "target": [0, 1]}
        )
        test_chronological_split(train, test, "time")

    def test_chronological_overlap_fails(self):
        train = pd.DataFrame(
            {"time": ["2026-01-01", "2026-01-03"], "target": [0, 1]}
        )
        test = pd.DataFrame(
            {"time": ["2026-01-02", "2026-01-04"], "target": [0, 1]}
        )
        with self.assertRaisesRegex(ValueError, "Chronological leakage"):
            test_chronological_split(train, test, "time")


if __name__ == "__main__":
    unittest.main()
