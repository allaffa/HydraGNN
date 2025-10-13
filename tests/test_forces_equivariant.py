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

import os
import pytest

import subprocess


@pytest.mark.parametrize("example", ["LennardJones"])
@pytest.mark.parametrize(
    "mpnn_type", ["SchNet", "EGNN", "DimeNet", "PAINN", "PNAPlus", "MACE"]
)
@pytest.mark.mpi_skip()
def pytest_examples(example, mpnn_type):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")  # Assuming different model scripts

    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    hydragnn_root = os.path.join(os.path.dirname(__file__), "..")
    env["PYTHONPATH"] = os.path.abspath(hydragnn_root)

    # Set environment variables to make tests faster for CI
    env["CI_MODE"] = "1"  # Signal to examples that we're in CI mode
    env["NUM_EPOCHS"] = "1"  # Use only 1 epoch for CI testing
    env["HYDRAGNN_VERBOSITY"] = "0"  # Reduce verbosity for faster execution

    return_code = subprocess.call(
        ["python", file_path, "--mpnn_type", mpnn_type],
        env=env,
        timeout=300,  # 5 minute timeout per test
    )

    # Check the file ran without error.
    assert return_code == 0
