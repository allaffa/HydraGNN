##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import torch
from hydragnn.utils.distributed import get_device_name
from torch.distributed.optim import ZeroRedundancyOptimizer

deepspeed_available = True
try:
    import deepspeed
except:
    deepspeed_available = False


def configure_aadl(optimizer, config):
    """Optionally attach AADL to an optimizer selected by HydraGNN.

    AADL wraps the optimizer's existing ``step`` method, so the optimizer type,
    state dict, scheduler, AMP scaler, and DDP gradient communication remain
    owned by PyTorch/HydraGNN.
    """
    aadl_config = config.get("AADL")
    if aadl_config is None:
        return optimizer
    if not isinstance(aadl_config, dict):
        raise TypeError("Optimizer.AADL must be an object")
    enabled = aadl_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("Optimizer.AADL.enabled must be a boolean")
    if not enabled:
        return optimizer
    options = {key: value for key, value in aadl_config.items() if key != "enabled"}
    # HydraGNN performs backward before optimizer.step and therefore cannot
    # supply AADL's reevaluation closure. Preserve that established loop and
    # make the lack of loss safeguarding explicit in the resulting config.
    if options.get("safeguard", False):
        raise ValueError(
            "HydraGNN AADL integration does not support safeguard=true because "
            "the training loop does not provide an optimizer closure"
        )
    options["safeguard"] = False
    try:
        import AADL
    except ImportError as error:
        raise ImportError(
            "Optimizer.AADL is enabled but AADL is not installed; "
            "install requirements-aadl.txt"
        ) from error
    AADL.accelerate(optimizer, **options)
    return optimizer


def select_standard_optimizer(model, config):
    optimizer = None

    if config["type"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "FusedLAMB":
        assert deepspeed_available, "deepspeed package not installed"
        assert (
            "cpu" != get_device_name()
        ), "GPUs not available to use FusedLAMB optimizer from deepspeed package"
        optimizer = deepspeed.ops.lamb.FusedLamb(
            model.parameters(), lr=config["learning_rate"]
        )
    else:
        raise NameError("The string used to identify the optimizer is NOT recognized")

    return optimizer


def select_zero_redundancy_optimizer(model, config):
    optimizer = None

    if config["type"] == "SGD":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.SGD,
            lr=config["learning_rate"],
        )
    elif config["type"] == "Adam":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=config["learning_rate"],
        )
    elif config["type"] == "Adadelta":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adadelta,
            lr=config["learning_rate"],
        )
    elif config["type"] == "Adagrad":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adagrad,
            lr=config["learning_rate"],
        )
    elif config["type"] == "Adamax":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adamax,
            lr=config["learning_rate"],
        )
    elif config["type"] == "AdamW":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=config["learning_rate"],
        )
    elif config["type"] == "RMSprop":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.RMSprop,
            lr=config["learning_rate"],
        )
    elif config["type"] == "FusedLAMB":
        assert deepspeed_available, "deepspeed package not installed"
        assert (
            "cpu" != get_device_name()
        ), "GPUs not available to use FusedLAMB optimizer from deepspeed package"
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=deepspeed.ops.lamb.FusedLamb,
            lr=config["learning_rate"],
        )
    else:
        raise NameError("The string used to identify the optimizer is NOT recognized")

    return optimizer


def select_optimizer(model, config):
    use_zero = False

    if "use_zero_redundancy" in config:
        use_zero = config["use_zero_redundancy"]

    if use_zero:
        optimizer = select_zero_redundancy_optimizer(model, config)
    else:
        optimizer = select_standard_optimizer(model, config)
    return configure_aadl(optimizer, config)
