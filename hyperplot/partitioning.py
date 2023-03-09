"""Module for Writing Partitioned Files."""

import os
from datetime import datetime

import polars as pl
import pandas as pd
from tsdownsample import LTTBDownsampler


def to_datetime(to_convert: int | float | str | datetime) -> datetime:
    """Convert to datetime from a given input.

    Args:
        to_convert: what should be a datetime

    Raises:
        ValueError: if the input cant be parsed correctly

    Returns:
        the converted input
    """
    if isinstance(to_convert, datetime):
        return to_convert

    elif isinstance(to_convert, (int, float)):
        return datetime.fromtimestamp(to_convert)

    elif isinstance(to_convert, str):
        try:
            return datetime.fromisoformat(to_convert)
        except ValueError:
            return datetime.strptime(to_convert, "%Y-%m-%d %H:%M:%S.%f")

    else:
        raise ValueError(f"Unrecognized format {to_convert}!")


def partition_df(
    df_or_path: str | pl.DataFrame | pd.DataFrame,
    partition_path: str,
    signal_name: str,
    hertz: int | float | None = None,
    levels: int = 10,
    max_points_per_partition: int = 100_000,
    linear_levels: bool = True,
) -> None:
    """Expects a dataframe with 2 columns, timestamp and value.

    Args:
        df_path: path of df (or df object) to read and partition, must contain only a timestamp and value column
        partition_path: path to top level partition, something like s3://my-bucket/. Can be any filesystem.
        signal_name: name of the signal (for pathing)
        hertz: frequency of the signal, if not provided will calculate.
        levels: Number of resolutions to store. Defaults to 20.
        linear_levels: use linear scale or log scale if False
    """
    if isinstance(df_or_path, pl.DataFrame):
        df = df_or_path
    elif isinstance(df_or_path, pd.DataFrame):
        df = pl.from_pandas(df_or_path)
    else:
        df = pl.read_csv(df_or_path) if df_or_path.endswith("csv") else pl.read_parquet(df_or_path)

    if df.columns != ["timestamp", "value"]:
        raise ValueError(f"Expected columns to be timestamp and value! Got {df.columns}.")

    if levels < 2:
        raise ValueError(f"'levels' must be > 1. Got {levels}.")

    # convert everything to UNIX time including nanoseconds
    # TODO: make sure this actually works for messy timestamps
    # Ideally this is written as nanoseconds so we can support high frequency stuff
    df = df.with_columns([pl.col("timestamp").apply(lambda x: to_datetime(x))])
    first = df["timestamp"].min().timestamp()
    last = df["timestamp"].max().timestamp()
    if hertz is None:
        hertz = int(round(len(df) / (last - first), 0))

    split_dfs = []
    start = 0
    end = max_points_per_partition
    while True:
        split_dfs.append(df[start:end, :])
        start = end
        end += max_points_per_partition

        if start >= len(df):
            break

    if partition_path.startswith(("s3", "gs")):
        makedirs = False
    else:
        sep = os.path.sep
        makedirs = True

    base_partition = sep.join([partition_path, signal_name])
    for split_df in split_dfs:
        split_start = str(int(split_df["timestamp"].min().timestamp()))
        split_path = sep.join([base_partition, split_start])
        if makedirs:
            os.makedirs(split_path, exist_ok=True)
        for level in range(1, levels + 1):
            if linear_levels:
                level_points = int(len(split_df) / level)
            else:
                if level == 1:
                    level_points = len(split_df)
                else:
                    level_points = int(len(split_df) / 10**level)
            x = split_df["timestamp"].to_numpy()
            y = split_df["value"].to_numpy()
            level_indices = LTTBDownsampler().downsample(x, y, n_out=level_points)
            level_df = pl.DataFrame(
                {
                    "timestamp": x[level_indices],
                    "value": y[level_indices],
                }
            )

            level_df.write_parquet(f"{split_path}{sep}{level}.parquet")
