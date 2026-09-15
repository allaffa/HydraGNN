##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hydragnn.utils.optimizer import configure_aadl, select_optimizer


@pytest.mark.parametrize(
    "optimizer_type",
    ["SGD", "Adam", "Adadelta", "Adagrad", "Adamax", "AdamW", "RMSprop"],
)
def test_aadl_wraps_every_standard_optimizer(optimizer_type):
    model = torch.nn.Linear(2, 1)
    calls = []
    fake_aadl = SimpleNamespace(
        accelerate=lambda optimizer, **options: calls.append((optimizer, options))
    )
    config = {
        "type": optimizer_type,
        "learning_rate": 0.01,
        "AADL": {
            "enabled": True,
            "acceleration_type": "anderson",
            "history_depth": 5,
        },
    }
    with patch.dict(sys.modules, {"AADL": fake_aadl}):
        optimizer = select_optimizer(model, config)
    assert calls == [(optimizer, {
        "acceleration_type": "anderson",
        "history_depth": 5,
        "safeguard": False,
    })]


def test_disabled_aadl_does_not_import_or_wrap():
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    assert configure_aadl(optimizer, {"AADL": {"enabled": False}}) is optimizer


def test_aadl_rejects_unavailable_loss_safeguard():
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    with pytest.raises(ValueError, match="does not support safeguard=true"):
        configure_aadl(optimizer, {"AADL": {"enabled": True, "safeguard": True}})


def test_aadl_configuration_must_be_an_object():
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    with pytest.raises(TypeError, match="must be an object"):
        configure_aadl(optimizer, {"AADL": True})
