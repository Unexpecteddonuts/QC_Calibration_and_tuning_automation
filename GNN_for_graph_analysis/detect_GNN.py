# The code loads a saved GNN model and uses it to detect dips in a given graph.
# The dips are defined as nodes with an output less than -0.5.

import torch
import matplotlib.pyplot as plt
from gnn import Net
from data import convert_matplotlib_to_GNN_format, load_graph_data

# Load the saved GNN model
model = Net()
model.load_state_dict(torch.load('/models/model.pt'))

# Load the graph from file
data_list = load_graph_data('/data/detect_graph.dat')
data = data_list[0]

# Detect dips in the graph
fig = plt.figure()
plt.plot(data.x.numpy(), data.y.numpy())
data = convert_matplotlib_to_GNN_format(fig)
out = model(data.x.view(-1, 1), data.edge_index, data.edge_attr)
dips = (out < -0.5).nonzero().squeeze().tolist()
print('Dips detected at nodes:', dips)