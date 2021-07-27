import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool
from .Base import Base


class PNNStack(Base):
    def __init__(
        self,
        deg: torch.Tensor,
        input_dim: int,
        output_dim: list,
        output_type: list,
        num_nodes: int,
        hidden_dim: int,
        config_heads: {},
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        ilossweights_hyperp: int = 1,  # if =1, considering weighted losses for different tasks and treat the weights as hyper parameters
        loss_weights: list = [1.0, 1.0, 1.0],  # weights for losses of different tasks
        ilossweights_nll: int = 0,  # if =1, using the scalar uncertainty as weights, as in paper
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    ):
        super().__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(
            PNAConv(
                in_channels=input_dim,
                out_channels=self.hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        )
        self.batch_norms.append(BatchNorm(self.hidden_dim))
        for _ in range(self.num_conv_layers - 1):
            conv = PNAConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))

        super()._multihead(output_dim, num_nodes, output_type, config_heads, ilossweights_hyperp, loss_weights, ilossweights_nll)

    def __str__(self):
        return "PNNStack"
