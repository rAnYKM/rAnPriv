# rAnPrivGP
Social Network Privacy based on SNAP Google Plus Dataset

## 1 Dataset Loader - snap_google.py
### Class SnapEgoNet:

####Important Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| root | String | The root node of ego network |
| featname | List | All feat names in the ego network |
| node | Dict | All neighboring nodes of the root node with their feature list|
| egofeat | List | The root node's feat list |
| edges | List | Edge list among neighboring nodes |
| follows | List | The root node's follow list |
| followers | List| The root node's follower list|


