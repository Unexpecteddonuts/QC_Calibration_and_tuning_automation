import torch
from torch_geometric.data import DataLoader
from gnn import Net
from data import generate_graph_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Train a GNN model on a small dataset
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
train_data = generate_graph_data(100, 10)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
for epoch in range(100):
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        out = model(data.x.view(-1, 1), data.edge_index, data.edge_attr)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('Epoch {}: Loss = {}'.format(epoch, total_loss / len(train_loader)))

# Visualize the node embeddings using t-SNE
node_embeddings = []
for data, target in train_data:
    out = model(data.x.view(-1, 1), data.edge_index, data.edge_attr)
    node_embeddings.append(out.detach().numpy())
node_embeddings = np.concatenate(node_embeddings, axis=0)
node_embeddings_tsne = TSNE(n_components=2).fit_transform(node_embeddings)
plt.scatter(node_embeddings_tsne[:, 0], node_embeddings_tsne[:, 1])
plt.show()
