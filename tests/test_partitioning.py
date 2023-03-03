"""Tests for partitioning."""

import pytest
from datetime import datetime, timedelta

from hyperplot import partitioning

import polars as pl
import numpy as np


@pytest.fixture()
def day_data() -> pl.DataFrame:
    """24 Hours of Data."""
    low = datetime(2023, 1, 1)
    high = datetime(2023, 1, 2)
    interval = timedelta(seconds=1)

    timestamp = pl.date_range(low, high, interval)
    value = np.sin(2 * np.pi * 1000 * np.arange(len(timestamp)) / len(timestamp))
