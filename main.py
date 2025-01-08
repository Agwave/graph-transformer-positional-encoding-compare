import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from sklearn.metrics import accuracy_score

from model import GraphTransformer


def run_train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    print(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def run_test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    return acc

def run():
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    data = dataset[0]
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"data: {data.x}")

    model = GraphTransformer(
        num_layers=3,
        in_features=dataset.num_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        num_heads=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = nn.NLLLoss()

    for epoch in range(200):
        loss = run_train(model, data, optimizer, criterion)
        if epoch % 10 == 0:
            acc = run_test(model, data)
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')


if __name__ == '__main__':
    run()
