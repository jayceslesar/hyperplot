"""Module for Writing Partitioned Files."""

import polars as pl
from tsdownsample import LTTBDownsampler


LEVELS = 20


def partition_df(df_path: str, partition_path: str, signal_name: str, hertz: int | float | None = None, levels: int = 20) -> None:
    """Expects a dataframe with 2 columns, timestamp and value.

    Args:
        df_path: path of df to read and partition, must contain only a timestamp and value column
        partition_path: path to top level partition, something like s3://my-bucket/. Can be any filesystem.
        signal_name: name of the signal (for pathing)
        hertz: frequency of the signal, if not provided will calculate.
        levels: Number of resolutions to store. Defaults to 20.
    """
    df = pl.read_csv(df_path) if df_path.endswith("csv") else pl.read_parquet(df_path)
    if df.columns != ["timestamp", "value"]:
        raise ValueError(f"Expected columns to be timestamp and value! Got {df.columns}.")

    if levels < 2:
        raise ValueError(f"'levels' must be > 1. Got {levels}.")

    # convert everything to UNIX time including microseconds
    df["timestamp"] = pl.Datetime()