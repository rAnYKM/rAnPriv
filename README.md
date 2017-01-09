# rAnPrivGP
Social Network Privacy based on SNAP Google Plus/Facebook Dataset

## Dataset Loader (Google+) - snap_google.py

### Class SnapEgoNet:

| Variable | Type | Description |
|-----------|------|-------------|
| root | String | The root node of ego network |
| featname | List | All feat names in the ego network |
| node | Dict | All neighboring nodes of the root node with their feature list|
| egofeat | List | The root node's feat list |
| edges | List | Edge list among neighboring nodes |
| follows | List | The root node's follow list |
| followers | List| The root node's follower list|

## Dataset Loader (Facebook) - snap_facebook.py

| Variable | Type | Description |
|-----------|------|-------------|
| root | String | The root node of ego network |
| featname | List | All feat names in the ego network |
| node | Dict | All neighboring nodes of the root node with their feature list|
| egofeat | List | The root node's feat list |
| edges | List | Edge list among neighboring nodes |
| friends | List | The root node's friend list |
| actor | Dict | Node Dict with path-like attribute access|
| category | rAnTree | Attribute category tree |
| ran | RanGraph | Ran Graph Network Collections |


## Relational Attribute Network (RAN) - ran_graph.py

### Class RanGraph

| Variable | Type | Description |
|--------|----|-----------|
| is_directed | Bool | Whether the social network is directed or undirected |
| soc_net | nx.Graph/nx.DiGraph | Default social network |
| soc_attr_net | nx.Graph/nx.DiGraph | Mixed social attribute network |
| attr_net | nx.Graph | Undirected attribute network |
| di_attr_net | nx.DiGraph | Directed attribute network |
