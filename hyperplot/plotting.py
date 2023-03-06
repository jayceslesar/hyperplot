"""Module for Plotting Partitioned Data."""

import os
from datetime import datetime, timezone

import dash
import dash_bootstrap_components as dbc
import fsspec
import plotly.graph_objects as go
import polars as pl
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State

external_stylesheets = [dbc.themes.BOOTSTRAP]


class HyperPlotter:
    """Class to Handle Plotting Partitioned Data."""

    def __init__(self, partition_path: str):
        self.partition_path = partition_path
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
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        app.layout = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Card(
                            children=[
                                dbc.CardHeader(html.H2("Channels")),
                                dcc.Checklist(
                                    self.channels,
                                    [],
                                    id="signals-checklist",
                                    labelStyle={"display": "block"},
                                    style={"height": 200, "width": 200, "overflow": "auto"},
                                ),
                            ],
                        )
                    ],
                    className="align-self-center",
                ),
                dbc.Row(
                    [
                        dbc.Card(
                            children=[
                                dbc.CardHeader(html.H2("Main Graph")),
                                dbc.CardBody(
                                    children=[
                                        dcc.Loading(
                                            children=[
                                                dcc.Graph(id="main-graph"),
                                            ]
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ]
                ),
            ]
        )

        @app.callback(
            Output("main-graph", "figure"),
            [Input("signals-checklist", "value"), Input("main-graph", "relayoutData")],
            prevent_initial_call=True,
        )
        def update_output(channels, relay_data):
            fig = go.Figure()
            if channels is None:
                return fig
            for channel in channels:
                paths, start, end = self._solve_partitions(channel, relay_data)
                df = pl.concat([pl.read_parquet(path) for path in paths])
                if start:
                    df = df.filter(pl.col("timestamp") >= datetime.fromtimestamp(start))
                if end:
                    df = df.filter(pl.col("timestamp") <= datetime.fromtimestamp(end))
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["value"], name=channel, mode="markers"))

            return fig

        self.app = app

    def get_channel_partitions(self, channel: str) -> list[str]:
        path = self.sep.join([self.partition_path, channel])
        return sorted([os.path.basename(partition) for partition in self.fs.ls(path)])

    def _solve_partitions(self, channel: str, relay_data: dict) -> tuple[list[str], datetime | None, datetime | None]:
        solution_paths = []
        partitions = self.get_channel_partitions(channel)
        if relay_data.get("autosize", None) or relay_data.get("xaxis.autorange", None):
            start = None
            end = None
            solution_partitions = partitions
        else:
            # :23 here because datetime cant handle it
            start = datetime.fromisoformat(relay_data["xaxis.range[0]"][:23]).timestamp()
            end = datetime.fromisoformat(relay_data["xaxis.range[1]"][:23]).timestamp()
            solution_partitions = [int(p) for p in partitions if start <= int(p) >= end]
            # case entirely inside a single partition...
            if not solution_partitions:
                solution_partitions = [[int(p) for p in partitions if int(p) >= start][-1]]
            else:  # get one before...
                one_before_idx = partitions.index(str(min(solution_partitions))) - 1
                solution_partitions = [partitions[one_before_idx]] + solution_partitions

        for solution_partition in solution_partitions:
            # this needs to be optimized by hertz somehow but we dont really care as we put 100k points in a partition as of now
            if len(solution_partitions) >= 10:
                level = 10
            else:
                level = len(solution_partitions)
            solution_path = self.sep.join([self.partition_path, channel, str(solution_partition), f"{level}.parquet"])
            solution_paths.append(solution_path)

        return solution_paths, start, end

    def serve(self) -> None:
        """Start the plotting instance with a default view."""
        self.app.run_server(debug=True)


# local demo working with 604,801 points for 2 channels
plotter = HyperPlotter("dataset")
plotter.serve()