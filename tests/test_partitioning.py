"""Tests for partitioning."""

import os
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from hyperplot.partitioning import partition_df


@pytest.fixture()
def week_data() -> pl.DataFrame:
    """1 week of 1hz Data."""
    low = datetime(2023, 1, 1)
    high = datetime(2023, 1, 8)
    interval = timedelta(seconds=1)

    timestamp = pl.date_range(low, high, interval)
    length = np.pi * 2 * 1000
    value = np.cos(np.arange(0, length, length / len(timestamp)))
    df = pl.DataFrame(
        {
            "timestamp": timestamp,
            "value": value,
        }
    )
    yield df


@pytest.mark.parametrize("df", ["week_data"])
def test_partition_df(df, request, tmpdir):
    df = request.getfixturevalue(df)
    signal_name = "test_signal"
    partition_path = str(tmpdir)
    outer_path = os.path.join(partition_path, signal_name)
    partition_df(df, partition_path, signal_name)
    assert len(os.listdir(outer_path)) == 7
    for path in os.listdir(outer_path):
        assert len(os.listdir(os.path.join(outer_path, path))) == 10
