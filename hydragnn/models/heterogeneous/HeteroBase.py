import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv,
    HGTConv,
    SAGEConv,
    GATConv,
    GCNConv,
    GINConv,
    PNAConv,
    global_mean_pool,
)
from torch_geometric.nn.models import MLP
from typing import Dict, List, Tuple, Optional


class HeteroBase(nn.Module):
    """Typed heterogeneous GNN base, paralleling Base for homogeneous graphs.

    Supports:
      - HeteroConv with per-edge-type conv choice (SAGE/GAT/GCN/GIN/PNA).
      - Optional hetero transformer stage via HGTConv.
      - Graph heads (pool-per-type then concatenate) and node heads per node type.
    """

    def __init__(
        self,
        metadata: Dict,
        hidden_dim: int,
        output_dim: List[int],
        output_type: List[str],
        node_output_map: Optional[List[str]] = None,
        conv_type: str = "SAGE",
        num_mp_layers: int = 3,
        transformer_cfg: Optional[Dict] = None,
        pna_deg: Optional[torch.Tensor] = None,
        loss_function: Optional[nn.Module] = None,
        loss_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.node_types: List[str] = metadata["node_types"]
        self.edge_types: List[Tuple[str, str, str]] = metadata["edge_types"]
        self.node_in_dims: Dict[str, int] = metadata["node_input_dims"]
        self.edge_in_dims: Dict[str, int] = metadata.get("edge_input_dims", {})
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_type = output_type
        self.node_output_map = node_output_map or [None] * len(output_type)
        self.loss_function = loss_function or F.mse_loss
        if loss_weights is None:
            loss_weights = [1.0] * len(output_dim)
        self.loss_weights = loss_weights
        self.conv_checkpointing = False

        self.node_encoders = nn.ModuleDict(
            {nt: nn.Linear(self.node_in_dims[nt], hidden_dim, bias=False) for nt in self.node_types}
        )
        self.edge_encoders = nn.ModuleDict()
        for _, rel, _ in self.edge_types:
            if rel in self.edge_in_dims:
                self.edge_encoders[rel] = nn.Linear(self.edge_in_dims[rel], hidden_dim, bias=False)

        self.graph_convs = nn.ModuleList()
        conv_type = conv_type.upper()
        for _ in range(num_mp_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                conv_dict[edge_type] = self._build_conv(conv_type, pna_deg)
            self.graph_convs.append(HeteroConv(conv_dict, aggr="sum"))

        transformer_cfg = transformer_cfg or {}
        self.use_transformer = transformer_cfg.get("enabled", False)
        self.transformer_layers = nn.ModuleList()
        if self.use_transformer:
            heads = transformer_cfg.get("heads", 4)
            t_layers = transformer_cfg.get("num_layers", 1)
            dropout = transformer_cfg.get("dropout", 0.1)
            for _ in range(t_layers):
                self.transformer_layers.append(
                    HGTConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        metadata=(self.node_types, self.edge_types),
                        heads=heads,
                        dropout=dropout,
                    )
                )

        self.graph_head = None
        if "graph" in output_type:
            pooled_dim = hidden_dim * len(self.node_types)
            first_graph_index = output_type.index("graph")
            self.graph_head = nn.Linear(pooled_dim, output_dim[first_graph_index])

        self.node_heads = nn.ModuleDict()
        for idx, ot in enumerate(output_type):
            if ot != "node":
                continue
            node_type = self.node_output_map[idx]
            if node_type is None:
                raise ValueError("node_output_map entry required for node head")
            if node_type not in self.node_heads:
                self.node_heads[node_type] = nn.Linear(hidden_dim, output_dim[idx])

    def _build_conv(self, conv_type: str, pna_deg: Optional[torch.Tensor]):
        if conv_type == "SAGE":
            return SAGEConv((-1, -1), self.hidden_dim)
        if conv_type == "GAT":
            return GATConv((-1, -1), self.hidden_dim, add_self_loops=False)
        if conv_type == "GCN":
            return GCNConv((-1, -1), self.hidden_dim)
        if conv_type == "GIN":
            mlp = MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim], act="relu", dropout=0.0)
            return GINConv(nn=mlp)
        if conv_type == "PNA":
            if pna_deg is None:
                raise ValueError("PNA conv requires pna_deg histogram")
            aggregators = ["mean", "min", "max", "std"]
            scalers = ["identity", "amplification", "attenuation"]
            return PNAConv(self.hidden_dim, self.hidden_dim, aggregators, scalers, pna_deg)
        raise ValueError(f"Unsupported conv_type: {conv_type}")

    def _encode_edges(self, data):
        edge_attr_dict = {}
        for edge_type in self.edge_types:
            et_data = data[edge_type]
            attr = getattr(et_data, "edge_attr", None)
            if attr is None:
                continue
            _, rel, _ = edge_type
            if rel in self.edge_encoders:
                edge_attr_dict[edge_type] = self.edge_encoders[rel](attr)
            else:
                edge_attr_dict[edge_type] = attr
        return edge_attr_dict

    def forward(self, data):
        x_dict = {nt: F.relu(self.node_encoders[nt](data[nt].x)) for nt in self.node_types}
        edge_attr_dict = self._encode_edges(data)

        for conv in self.graph_convs:
            x_prev = x_dict
            x_out = conv(
                x_prev,
                data.edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )
            # Keep representations for node types that are not destinations in this layer.
            for nt in self.node_types:
                if nt not in x_out:
                    x_out[nt] = x_prev[nt]
            x_dict = {nt: F.relu(x_out[nt]) for nt in self.node_types}

        if self.use_transformer:
            for hgt in self.transformer_layers:
                x_prev = x_dict
                x_out = hgt(
                    x_prev,
                    data.edge_index_dict,
                    edge_attr_dict=edge_attr_dict,
                )
                for nt in self.node_types:
                    if nt not in x_out:
                        x_out[nt] = x_prev[nt]
                x_dict = {nt: F.relu(x_out[nt]) for nt in self.node_types}

        outputs = []
        for head_idx, head_type in enumerate(self.output_type):
            if head_type == "graph":
                pooled = [global_mean_pool(x_dict[nt], data[nt].batch) for nt in self.node_types]
                graph_feat = torch.cat(pooled, dim=-1)
                outputs.append(self.graph_head(graph_feat))
            elif head_type == "node":
                node_type = self.node_output_map[head_idx]
                head = self.node_heads[node_type]
                outputs.append(head(x_dict[node_type]))
            else:
                raise ValueError(f"Unknown head type: {head_type}")
        return outputs

    def loss(self, pred, target, head_index):
        tot_loss = 0
        tasks_loss = []
        for ihead, head_pre in enumerate(pred):
            head_val = target[head_index[ihead]]
            if head_pre.shape != head_val.shape:
                head_val = torch.reshape(head_val, head_pre.shape)
            task_loss = self.loss_function(head_pre, head_val)
            tasks_loss.append(task_loss)
            tot_loss += task_loss * self.loss_weights[ihead]
        return tot_loss, tasks_loss, []

    def enable_conv_checkpointing(self):
        # Placeholder to keep parity with Base stacks; no checkpointing wiring needed here.
        self.conv_checkpointing = True
