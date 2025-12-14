import torch
from torch_geometric.data import HeteroData

from hydragnn.models.heterogeneous import HeteroBase


def _toy_data():
    data = HeteroData()
    num_papers = 3
    num_authors = 2

    data["paper"].x = torch.randn(num_papers, 8)
    data["paper"].batch = torch.zeros(num_papers, dtype=torch.long)
    data["author"].x = torch.randn(num_authors, 4)
    data["author"].batch = torch.zeros(num_authors, dtype=torch.long)

    data[("paper", "cites", "paper")].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    data[("author", "writes", "paper")].edge_index = torch.tensor(
        [[0, 1], [0, 2]], dtype=torch.long
    )

    data["paper"].y = torch.randn(num_papers, 4)
    data.graph_y = torch.randn(1, 1)
    return data


def pytest_hetero_stack_forward_and_loss():
    metadata = {
        "node_types": ["paper", "author"],
        "edge_types": [("paper", "cites", "paper"), ("author", "writes", "paper")],
        "node_input_dims": {"paper": 8, "author": 4},
        "edge_input_dims": {},
    }

    model = HeteroBase(
        metadata=metadata,
        hidden_dim=16,
        output_dim=[1, 4],
        output_type=["graph", "node"],
        node_output_map=[None, "paper"],
        conv_type="SAGE",
        num_mp_layers=2,
        transformer_cfg={"enabled": False},
    )

    data = _toy_data()
    pred = model(data)
    assert len(pred) == 2
    assert pred[0].shape == (1, 1)
    assert pred[1].shape[0] == data["paper"].x.size(0)
    assert pred[1].shape[1] == 4

    target = [data.graph_y, data["paper"].y]
    head_index = [0, 1]
    total_loss, task_losses, _ = model.loss(pred, target, head_index)
    assert torch.isfinite(total_loss)
    assert len(task_losses) == 2
