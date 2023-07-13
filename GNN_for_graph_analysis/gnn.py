import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_max_pool
from torch_geometric.nn import global_mean_pool, global_add_pool

class Net(torch.nn.Module):
    def __init__(self, conv_type='GCN', hidden_channels=64, num_layers=4, dropout=0.5):
        super(Net, self).__init__()
        self.conv_type = conv_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        if conv_type == 'GCN':
            self.conv = GCNConv
        elif conv_type == 'GAT':
            self.conv = GATConv
        elif conv_type == 'SAGE':
            self.conv = SAGEConv
        elif conv_type == 'GIN':
            self.conv = GINConv

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(self.conv(1, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for i in range(num_layers - 1):
            self.convs.append(self.conv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
        self.dropout_layer = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        x = global_mean_pool(x, edge_index)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.lin2(x)
        return x

    def l1_regularization(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, p=1)
        return l1_reg

    def l2_regularization(self):
        l2_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg

    def forward_with_pooling(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        x = global_max_pool(x, edge_index)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.lin2(x)
        return x

    def forward_with_residual(self, x, edge_index):
        out = x
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
            x = x + out
            out = x
        x = global_mean_pool(x, edge_index)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.lin2(x)
        return x

    def forward_with_l1_regularization(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        x = global_mean_pool(x, edge_index)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.lin2(x)
        l1_reg = self.l1_regularization()
        return x + l1_reg

    def forward_with_l2_regularization(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        x = global_mean_pool(x, edge_index)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.lin2(x)
        l2_reg = self.l2_regularization()
        return x + l2_reg
