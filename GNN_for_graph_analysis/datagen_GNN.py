import torch
from torch_geometric.data import DataLoader
from gnn import Net
from data import load_graph_data

# Load the data from file
data_list = load_graph_data('/data/trainingdata.dat')

# Pre-train a GNN model on a large dataset
pretrained_model = Net()
pretrained_optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=0.01)
pretrained_criterion = torch.nn.MSELoss()
pretrained_train_loader = DataLoader(data_list[:900], batch_size=10, shuffle=True)
for epoch in range(100):
    total_loss = 0
    for data in pretrained_train_loader:
        pretrained_optimizer.zero_grad()
        out = pretrained_model(data.x.view(-1, 1), data.edge_index, data.edge_attr)
        loss = pretrained_criterion(out, data.y)
        loss.backward()
        pretrained_optimizer.step()
        total_loss += loss.item()
    print('Pre-training Epoch {}: Loss = {}'.format(epoch, total_loss / len(pretrained_train_loader)))

# Freeze the weights of the pre-trained model
for param in pretrained_model.parameters():
    param.requires_grad = False

# Fine-tune the pre-trained model on a small dataset
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
train_loader = DataLoader(data_list[900:], batch_size=10, shuffle=True)
for epoch in range(100):
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = pretrained_model(data.x.view(-1, 1), data.edge_index, data.edge_attr)
        out = model(out, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('Fine-tuning Epoch {}: Loss = {}'.format(epoch, total_loss / len(train_loader)))
    
# Save the trained model
torch.save(model.state_dict(), '/models/model.pt')