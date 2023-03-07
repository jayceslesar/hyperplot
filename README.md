# Hyperplot

## Lets plot a lot of data!

Tool for efficiently writing partitioned time series data to some file-backend and serving that data in plotly.

Robust to out of order data but not duplicates so best to ingest data on an event.


### Open Tasks:

- Async Read Requests

- Timestamps should be in nanoseconds -> convert on selection to find partitions to read

- Better level solver


### Open Questions:

- Optimal Partition Size?

- Linear or Log levels?
