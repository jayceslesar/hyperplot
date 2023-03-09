"""Module for Plotting Partitioned Data."""

import os
from datetime import datetime, timezone

import dash
import dash_bootstrap_components as dbc
import fsspec
import numpy as np
import plotly.graph_objects as go
import polars as pl
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
from tsdownsample import LTTBDownsampler

from hyperplot.partitioning import to_datetime

external_stylesheets = [dbc.themes.BOOTSTRAP]


class HyperPlotter:
    """Class to Handle Plotting Partitioned Data."""

    def __init__(self, dataset_paths: dict[str, str], max_level: int, max_points: int = 5_000, points_cutoff: int = 100):
        self.dataset_paths = dataset_paths
        self.max_level = max_level
        self.max_points = max_points
        self.points_cutoff = points_cutoff
        # store previous zooms
        self.previous_starts = set()
        self.previous_ends = set()

        # figure out what filesystem to use
        dataset_loc = self.dataset_paths[list(self.dataset_paths.keys())[0]]
        if dataset_loc.startswith("s3"):
            self.fs = fsspec.filesystem("s3")
            self.sep = "/"
        elif dataset_loc.startswith("gs"):
            self.fs = fsspec.filesystem("gs")
            self.sep = "/"
        else:
            self.fs = fsspec.filesystem("file")
            self.sep = os.path.sep

        self.channels = {}
        for dataset in self.dataset_paths:
            self.channels[dataset] = sorted(
                [os.path.basename(signal) for signal in self.fs.ls(self.dataset_paths[dataset])]
            )

        dropdowns = []
        self.inputs = []
        vh = 100 / len(self.dataset_paths)
        for dataset in self.dataset_paths:
            input_name = f"{dataset}-signals-checklist"
            self.inputs.append(Input(input_name, "value"))
            dropdown = dbc.DropdownMenu(
                children=[
                    dcc.Checklist(
                        self.channels[dataset],
                        [],
                        id=input_name,
                        labelStyle={"display": "block"},
                        style={"height": f"{vh}vh", "overflow": "auto"},
                    )
                ],
                label=dataset,
            )
            dropdowns.append(dropdown)
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        app.layout = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    children=[
                                        dbc.CardHeader(html.H2("Channels")),
                                        *dropdowns,
                                    ],
                                    style={"height": "100vh"},
                                )
                            ],
                            className="align-self-center",
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    children=[
                                        dbc.CardHeader(html.H2("Plotting")),
                                        dbc.CardBody(
                                            children=[
                                                dcc.Loading(
                                                    children=[
                                                        dcc.Graph(id="main-graph", style={"height": "90vh"}),
                                                    ]
                                                )
                                            ],
                                        ),
                                    ]
                                ),
                            ],
                            width=9,
                        ),
                    ],
                )
            ]
        )

        @app.callback(
            Output("main-graph", "figure"),
            [*self.inputs, Input("main-graph", "relayoutData")],
            prevent_initial_call=True,
        )
        def update_output(*args):
            traces = {}
            dataset_selections = args[:-1]
            relay_data = args[-1]
            if not any(dataset_selections):
                return go.Figure()
            for channels in dataset_selections:
                for channel in channels:
                    paths, start, end = self._solve_partitions(channel, relay_data)
                    # this can be threaded for sure
                    df = pl.concat([pl.read_parquet(path) for path in paths]).sort("timestamp")
                    if start:
                        df = df.filter(pl.col("timestamp") >= datetime.fromtimestamp(start))
                    if end:
                        df = df.filter(pl.col("timestamp") <= datetime.fromtimestamp(end))

                    x = df["timestamp"]
                    y = df["value"]
                    if len(df) > self.max_points:
                        x = df["timestamp"].to_numpy()
                        y = df["value"].to_numpy()
                        indices = LTTBDownsampler().downsample(x, y, n_out=self.max_points)
                        df = pl.DataFrame(
                            {
                                "timestamp": x[indices],
                                "value": y[indices],
                            }
                        ).sort("timestamp")
                        x = list(df["timestamp"])
                        y = list(df["value"])

                        # calculate draw lines or not
                        mean_delta = np.mean(np.diff(x))
                        gaps_x = [x[0]]
                        gaps_y = [y[0]]
                        line_diff_cutoff = 5 * mean_delta

                        for i in range(len(df)):
                            if abs(x[i] - gaps_x[-1]) > line_diff_cutoff:
                                # interpolate timestamp with middle value and add None so we see a line break
                                gaps_x.append(x[i] + (gaps_x[-1] - x[i]) / 2)
                                gaps_y.append(None)
                            gaps_x.append(x[i])
                            gaps_y.append(y[i])
                        x = gaps_x
                        y = gaps_y

                    # always draw markers less than self.points_cutoff
                    if len(y) <= self.points_cutoff:
                        mode = "markers"
                        marker = {"size": 5}
                        line = None
                    else:
                        mode = "lines"
                        marker = None
                        line = {"width": 2}

                    # magnitude calc so we can create subplots for traces with very different scales
                    real_values = [val for val in y if val is not None]
                    real_values = [val for val in real_values if val != float("nan")]
                    real_values = [val for val in real_values if val != np.nan]
                    real_values = [val for val in real_values if str(val) != "nan"]
                    min_y = abs(min(real_values))
                    max_y = abs(max(real_values))
                    magnitude = len(str(int(sum((min_y, max_y)))))
                    # bin these together because they are small enough
                    magnitude = 1 if magnitude == 2 else magnitude
                    if magnitude not in traces:
                        traces[magnitude] = []
                    traces[magnitude].append(
                        go.Scatter(
                            x=x,
                            y=y,
                            name=channel,
                            mode=mode,
                            marker=marker,
                            line=line,
                            showlegend=True,
                            legendgroup=magnitude,
                        )
                    )

            if not traces:
                return go.Figure()

            fig = make_subplots(rows=len(traces), cols=1, shared_xaxes=True)
            for i, magnitude in enumerate(traces):
                magnitude_traces = traces[magnitude]
                for trace in magnitude_traces:
                    fig.add_trace(trace, row=i + 1, col=1)

            fig.update_layout(legend_tracegroupgap=80)
            return fig

        self.app = app

    def get_channel_partitions(self, channel: str) -> tuple[str, list[str]]:
        for dataset in self.channels:
            if channel in self.channels[dataset]:
                partition_path = self.dataset_paths[dataset]
        dataset = [dataset for dataset in self.dataset_paths]
        path = self.sep.join([partition_path, channel])
        return partition_path, sorted([os.path.basename(partition) for partition in self.fs.ls(path)])

    def _solve_partitions(self, channel: str, relay_data: dict) -> tuple[list[str], datetime | None, datetime | None]:
        solution_paths = []
        partition_path, partitions = self.get_channel_partitions(channel)
        if (
            relay_data.get("autosize", None)
            or relay_data.get("xaxis.autorange", None)
            or "xaxis.range[0]" not in relay_data
        ):
            start = None
            end = None
            solution_partitions = partitions
        else:
            start = to_datetime(relay_data["xaxis.range[0]"]).timestamp()
            end = to_datetime(relay_data["xaxis.range[1]"]).timestamp()
            self.previous_starts.add(relay_data["xaxis.range[0]"])
            self.previous_ends.add(relay_data["xaxis.range[1]"])

            solution_partitions = [int(p) for p in partitions if start <= int(p) <= end]
            # case entirely inside a single partition...
            if not solution_partitions:
                if len(partitions) == 1:
                    solution_partitions = partitions
                else:
                    solution_partitions = [[int(p) for p in partitions if int(p) >= start][0]]

            one_before_idx = partitions.index(str(min(solution_partitions))) - 1
            solution_partitions = [partitions[one_before_idx]] + solution_partitions

        for solution_partition in solution_partitions:
            # this needs to be optimized by hertz somehow but we dont really care as we put 100k points in a partition as of now
            if len(solution_partitions) >= self.max_level:
                level = self.max_level
            else:
                level = len(solution_partitions)
            solution_path = self.sep.join([partition_path, channel, str(solution_partition), f"{level}.parquet"])
            solution_paths.append(solution_path)

        return solution_paths, start, end

    def serve(self) -> None:
        """Start the plotting instance with a default view."""
        self.app.run_server(debug=True, host="0.0.0.0")
