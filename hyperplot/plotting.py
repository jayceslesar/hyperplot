"""Module for Plotting Partitioned Data."""

import os
import fsspec
import plotly.graph_objects as go



class PartitionedPlotter:
    """Class to Handle Plotting Partitioned Data."""

    def __init__(self, partition_path: str):
        self.partition_path = partition_path
        # print(fsspec.glob(self.partition_path))
        if self.partition_path.startswith("s3"):
             self.fs = fsspec.filesystem("s3")
             self.sep = "/"
        elif self.partition_path.startswith("gs"):
             self.fs = fsspec.filesystem("gs")
             self.sep = "/"
        else:
            self.fs = fsspec.filesystem("file")
            self.sep = os.path.sep

        self.channels = [os.path.basename(signal) for signal in self.fs.ls(self.partition_path)]

    def get_channel_partitions(self, channel: str) -> list[str]:
        path = self.sep.join([self.partition_path, channel])
        return [os.path.basename(partition) for partition in self.fs.ls(path)]

    def serve(self) -> None:
        pass
