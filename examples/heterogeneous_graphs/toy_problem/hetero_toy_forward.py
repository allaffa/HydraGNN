"""Minimal smoke test for HeteroStack.

Run:
    python examples/heterogeneous_graphs/toy_problem/hetero_toy_forward.py
"""

import torch
from torch_geometric.data import HeteroData

from hydragnn.models.heterogeneous import HeteroBase


def build_toy_data():
    data = HeteroData()
    num_papers = 4
    num_authors = 3

    data["paper"].x = torch.randn(num_papers, 8)
    data["paper"].batch = torch.zeros(num_papers, dtype=torch.long)
    data["author"].x = torch.randn(num_authors, 4)
    data["author"].batch = torch.zeros(num_authors, dtype=torch.long)

    cites_edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data[("paper", "cites", "paper")].edge_index = cites_edges
    data[("paper", "cites", "paper")].edge_attr = torch.randn(cites_edges.size(1), 2)

    writes_edges = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    data[("author", "writes", "paper")].edge_index = writes_edges
    data[("author", "writes", "paper")].edge_attr = torch.randn(writes_edges.size(1), 1)

    data["paper"].y = torch.randn(num_papers, 4)
    data["author"].y = torch.randn(num_authors, 2)
    data.graph_y = torch.randn(1, 1)
    return data


def main():
    hetero_config = {
        "node_types": ["paper", "author"],
        "edge_types": [["paper", "cites", "paper"], ["author", "writes", "paper"]],
        "node_input_dims": {"paper": 8, "author": 4},
        "edge_input_dims": {"cites": 2, "writes": 1},
        "node_output_map": [None, "paper"],
        "conv_type": "SAGE",
        "num_mp_layers": 2,
        "transformer": {"enabled": True, "num_layers": 1, "heads": 2, "dropout": 0.1},
    }

    metadata = {
        "node_types": hetero_config["node_types"],
        "edge_types": [tuple(et) for et in hetero_config["edge_types"]],
        "node_input_dims": hetero_config["node_input_dims"],
        "edge_input_dims": hetero_config.get("edge_input_dims", {}),
    }

    model = HeteroBase(
        metadata=metadata,
        hidden_dim=32,
        output_dim=[1, 4],
        output_type=["graph", "node"],
        node_output_map=hetero_config["node_output_map"],
        conv_type=hetero_config["conv_type"],
        num_mp_layers=hetero_config["num_mp_layers"],
        transformer_cfg=hetero_config["transformer"],
    )

    data = build_toy_data()
    pred = model(data)
    print("Graph head shape:", pred[0].shape)
    print("Paper node head shape:", pred[1].shape)

    target = [data.graph_y, data["paper"].y]
    head_index = [0, 1]
    total_loss, task_losses, _ = model.loss(pred, target, head_index)
    print("Loss:", float(total_loss))
    print("Task losses:", [float(x) for x in task_losses])


if __name__ == "__main__":
    main()
