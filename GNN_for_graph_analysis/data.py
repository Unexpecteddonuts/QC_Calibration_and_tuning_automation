import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_graph_data(filename):
    with open(filename, 'rb') as f:
        figs = pickle.load(f)
    data_list = []
    for fig in figs:
        data = convert_matplotlib_to_GNN_format(fig)
        data_list.append(data)
    with open('/data/trainingdata.dat', 'wb') as f:
        pickle.dump(data_list, f)
    return data_list

def convert_matplotlib_to_GNN_format(fig):
    ax = fig.gca()
    G = nx.Graph()
    for line in ax.lines:
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        for i in range(len(xdata) - 1):
            G.add_edge(xdata[i], ydata[i], weight=ydata[i+1])
    adj = nx.adjacency_matrix(G).todense()
    x = torch.tensor(np.random.rand(adj.shape[0], 1), dtype=torch.float)
    edge_index = torch.tensor(np.array(G.edges()).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(G.edges(data='weight'))[:, 2].reshape(-1, 1), dtype=torch.float)
    y = torch.tensor(np.array([nx.is_chordal(G)], dtype=np.float32))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data