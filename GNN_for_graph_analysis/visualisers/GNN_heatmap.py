import torch
import matplotlib.pyplot as plt
from gnn import Net
from data import convert_matplotlib_to_GNN_format

# Load the saved GNN model
model = Net()
model.load_state_dict(torch.load('../models/model.pt'))

# Generate a heatmap of the node features
fig = plt.figure()
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 4, 3, 2, 1, 2])
data = convert_matplotlib_to_GNN_format(fig)
out = model(data.x.view(-1, 1), data.edge_index, data.edge_attr)
node_features = out.detach().numpy().squeeze()
plt.scatter(range(1, 11), node_features, c=node_features, cmap='coolwarm')
plt.colorbar()
plt.show()
