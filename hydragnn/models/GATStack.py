##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import pdb
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import ReLU, Linear, Identity
from torch_geometric.nn import GATv2Conv, BatchNorm, Sequential

from .Base import Base


class GATStack(Base):
    def __init__(
        self,
        input_args,
        conv_args,
        heads: int,
        negative_slope: float,
        edge_dim: int,
        *args,
        **kwargs,
    ):
        # note that self.heads is a parameter in GATConv, not the num_heads in the output part
        self.heads = heads
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim
        self.is_edge_model = True  # specify that mpnn can handle edge features
        super().__init__(input_args, conv_args, *args, **kwargs)

    def _init_conv(self):
        """Here this function overwrites _init_conv() in Base since it has different implementation
        in terms of dimensions due to the multi-head attention"""
        if self.use_global_attn:
            self.graph_convs.append(
                self._apply_global_attn(
                    self.get_conv(
                        self.embed_dim,
                        self.hidden_dim,
                        concat=True,
                        edge_dim=self.edge_embed_dim,
                    )
                )
            )
            self.feature_layers.append(BatchNorm(self.hidden_dim))
            for _ in range(self.num_conv_layers - 2):
                self.graph_convs.append(
                    self._apply_global_attn(
                        self.get_conv(
                            self.hidden_dim,
                            self.hidden_dim,
                            concat=True,
                            edge_dim=self.edge_embed_dim,
                        )
                    )
                )
                self.feature_layers.append(BatchNorm(self.hidden_dim))
            self.graph_convs.append(
                self._apply_global_attn(
                    self.get_conv(
                        self.hidden_dim,
                        self.hidden_dim,
                        concat=False,
                        edge_dim=self.edge_embed_dim,
                    )
                )
            )
            self.feature_layers.append(BatchNorm(self.hidden_dim))
        else:
            self.graph_convs.append(
                self._apply_global_attn(
                    self.get_conv(
                        self.embed_dim,
                        self.hidden_dim,
                        concat=True,
                        edge_dim=self.edge_embed_dim,
                    )
                )
            )
            self.feature_layers.append(BatchNorm(self.hidden_dim * self.heads))
            for _ in range(self.num_conv_layers - 2):
                self.graph_convs.append(
                    self._apply_global_attn(
                        self.get_conv(
                            self.hidden_dim * self.heads,
                            self.hidden_dim,
                            concat=True,
                            edge_dim=self.edge_embed_dim,
                        )
                    )
                )
                self.feature_layers.append(BatchNorm(self.hidden_dim * self.heads))
            self.graph_convs.append(
                self._apply_global_attn(
                    self.get_conv(
                        self.hidden_dim * self.heads,
                        self.hidden_dim,
                        concat=False,
                        edge_dim=self.edge_embed_dim,
                    )
                )
            )
            self.feature_layers.append(BatchNorm(self.hidden_dim))

    def _init_node_conv(self):
        """Here this function overwrites _init_conv() in Base since it has different implementation
        in terms of dimensions due to the multi-head attention"""
        # *******convolutional layers for node level predictions*******#
        # two ways to implement node features from here:
        # 1. one graph for all node features
        # 2. one graph for one node features (currently implemented)
        nodeconfiglist = self.config_heads["node"]
        assert self.num_branches == len(
            nodeconfiglist
        ), "asumming node head has the same branches as graph head, if any"
        for branchdict in nodeconfiglist:
            # only support conv for all node branches
            if branchdict["architecture"]["type"] != "conv":
                return

        node_feature_ind = [
            i for i, head_type in enumerate(self.head_type) if head_type == "node"
        ]
        if len(node_feature_ind) == 0:
            return

        for branchdict in nodeconfiglist:
            branchtype = branchdict["type"]
            brancharct = branchdict["architecture"]
            num_conv_layers_node = brancharct["num_headlayers"]
            hidden_dim_node = brancharct["dim_headlayers"]

            convs_node_hidden = ModuleList()
            batch_norms_node_hidden = ModuleList()
            convs_node_output = ModuleList()
            batch_norms_node_output = ModuleList()

            # In this part, each head has same number of convolutional layers, but can have different output dimension
            convs_node_hidden.append(
                self.get_conv(self.hidden_dim, hidden_dim_node[0], True)
            )
            batch_norms_node_hidden.append(BatchNorm(hidden_dim_node[0] * self.heads))
            for ilayer in range(num_conv_layers_node - 1):
                convs_node_hidden.append(
                    self.get_conv(
                        hidden_dim_node[ilayer] * self.heads,
                        hidden_dim_node[ilayer + 1],
                        True,
                    )
                )
                batch_norms_node_hidden.append(
                    BatchNorm(hidden_dim_node[ilayer + 1] * self.heads)
                )
            for ihead in node_feature_ind:
                convs_node_output.append(
                    self.get_conv(
                        hidden_dim_node[-1] * self.heads, self.head_dims[ihead], False
                    )
                )
                batch_norms_node_output.append(BatchNorm(self.head_dims[ihead]))

            self.convs_node_hidden[branchtype] = convs_node_hidden
            self.batch_norms_node_hidden[branchtype] = batch_norms_node_hidden
            self.convs_node_output[branchtype] = convs_node_output
            self.batch_norms_node_output[branchtype] = batch_norms_node_output

    def get_conv(self, input_dim, output_dim, concat, edge_dim=None):
        gat = GATv2Conv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=self.heads,
            negative_slope=self.negative_slope,
            dropout=self.dropout,
            add_self_loops=True,
            edge_dim=edge_dim,
            concat=concat,
        )

        if self.use_global_attn and concat:
            self.out_lin = Linear(self.hidden_dim * self.heads, self.hidden_dim)
        else:
            self.out_lin = Identity()

        return Sequential(
            self.input_args,
            [
                (gat, self.conv_args + " -> inv_node_feat"),
                (self.out_lin, "inv_node_feat -> inv_node_feat"),
                (
                    lambda inv_node_feat, equiv_node_feat: [
                        inv_node_feat,
                        equiv_node_feat,
                    ],
                    "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                ),
            ],
        )

    def __str__(self):
        return "GATStack"
