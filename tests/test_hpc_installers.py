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

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
INSTALLERS = (
    ROOT / "scripts/hpc/olcf/frontier/installation/install.sh",
    ROOT / "scripts/hpc/alcf/aurora/installation/install.sh",
    ROOT / "scripts/hpc/nersc/perlmutter/installation/install.sh",
)


def test_hpc_installers_are_valid_bash():
    for installer in INSTALLERS:
        subprocess.run(["bash", "-n", installer], check=True)


def test_hpc_installers_install_hydragnn_before_fairchem():
    for installer in INSTALLERS:
        script = installer.read_text()
        hydragnn_install = script.index("--no-deps -e")
        fairchem_install = script.index(
            '"fairchem-core==${FAIRCHEM_CORE_VERSION}"'
        )
        assert hydragnn_install < fairchem_install
