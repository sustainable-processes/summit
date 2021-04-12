import pytest
from summit import DataSet


def test_dataset():
    data_columns = ["tau", "equiv_pldn", "conc_dfnb", "temperature"]
    metadata_columns = ["strategy"]
    columns = data_columns + metadata_columns

    # Using arrays
    values = [[1.5, 0.5, 0.1, 30.0, "test"]]
    ds = DataSet(values, columns=columns, metadata_columns="strategy")
    assert all(ds.columns.get_level_values("NAME").tolist()) == all(columns)
    assert all(ds.data_columns) == all(data_columns)
    assert all(ds.metadata_columns) == all(metadata_columns)

    # Test creating datset with dictionary
    values = {
        ("tau", "DATA"): [1.5, 10.0],
        ("equiv_pldn", "DATA"): [0.5, 3.0],
        ("conc_dfnb", "DATA"): [0.1, 4.0],
        ("temperature", "DATA"): [30.0, 100.0],
        ("strategy", "METADATA"): ["test", "test"],
    }
    ds = DataSet(values)
    assert all(ds.columns.get_level_values("NAME").tolist()) == all(columns)
    assert all(ds.data_columns) == all(data_columns)
    assert all(ds.metadata_columns) == all(metadata_columns)
