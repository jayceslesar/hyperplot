# Hyperplot

## Lets plot a lot of data!

Tool for efficiently writing partitioned time series data to some file-backend and serving that data in plotly.

Robust to out of order data but not duplicates so best to ingest data on an event.


### Open Tasks:

- Async Read Requests

- Timestamps should be in nanoseconds -> convert on selection to find partitions to read

- Better level solver

- Only redraw graph if partitions have changed

- Smart line drawing (when do we show lines vs points)


### Open Questions:

- Optimal max points in highest resolution?

- Linear or log steps in levels?
