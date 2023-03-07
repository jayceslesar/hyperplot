# Hyperplot

## Lets plot a lot of data!

Tool for efficiently writing partitioned time series data to some file-backend and serving that data in plotly.

Robust to out of order data but not duplicates so best to ingest data on an event.


### Open Tasks:

- Async Read Requests

- Timestamps should be in nanoseconds -> convert on selection to find partitions to read

- Better level solver

- Only redraw graph if partitions have changed

- Can unclick only one trace in a subplot where there may be multiple traces on one subplot (have to use the checkboxes)

- Issue where a trace could change subplots when magnitudes are recalculated (maybe okay?)


### Open Questions:

- Optimal max points in highest resolution?


