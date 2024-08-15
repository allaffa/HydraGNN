##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import json, os
import sys
import logging
import pickle
from tqdm import tqdm
from mpi4py import MPI
import argparse

import torch
import numpy as np

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.distributed import get_device
from hydragnn.utils.print_utils import print_distributed
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.config_utils import (
    update_config,
)
from hydragnn.models.create import create_model_config
from hydragnn.preprocess import create_dataloaders

from scipy.interpolate import griddata

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import matplotlib.pyplot as plt
from ensemble_utils import model_ensemble, test_ens

plt.rcParams.update({"font.size": 20})


def getcolordensity(xdata, ydata):
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
    return hist2d_norm


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":

    modelname = "MO2"

    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir_folder", help="folder of trained models", type=str, default=None)
    parser.add_argument("--log", help="log name", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pickle",
        help="Pickle gan_dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    modeldirlist = [os.path.join(args.models_dir_folder, name) for name in os.listdir(args.models_dir_folder) if os.path.isdir(os.path.join(args.models_dir_folder, name))]

    var_config = None
    for modeldir in modeldirlist:
        input_filename = os.path.join(modeldir, "config.json")
        with open(input_filename, "r") as f:
            config = json.load(f)
        if var_config is not None:
            assert var_config==config["NeuralNetwork"]["Variables_of_interest"], "Inconsistent variable config in %s"%input_filename
        else:
            var_config = config["NeuralNetwork"]["Variables_of_interest"]
    verbosity=config["Verbosity"]["level"]
    log_name = "GFM_EnsembleInference" if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD

    comm.Barrier()

    timer = Timer("load_data")
    timer.start()
    if args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(
            basedir=basedir,
            label="trainset",
            var_config=config["NeuralNetwork"]["Variables_of_interest"],
        )
        valset = SimplePickleDataset(
            basedir=basedir,
            label="valset",
            var_config=config["NeuralNetwork"]["Variables_of_interest"],
        )
        testset = SimplePickleDataset(
            basedir=basedir,
            label="testset",
            var_config=config["NeuralNetwork"]["Variables_of_interest"],
        )
        pna_deg = trainset.pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))
    ##################################################################################################################
    model_ens = model_ensemble(modeldirlist)
    model_ens = hydragnn.utils.get_distributed_model(model_ens, verbosity)
    model_ens.eval()
    ##################################################################################################################
    #model inference for a single graph
    device = next(model_ens.parameters()).device
    graphdata=testset[0].to(device)
    ytrue = graphdata.y
    with torch.no_grad():
        pred_mean, pred_std=model_ens(graphdata, meanstd=True) 
    ind = 0
    print("For graph:", graphdata)
    print("Energy, True VS Pred at Uncertainty: ", ytrue[0].item(), pred_mean[0].item(), pred_std[0].item())
    num_nodes=graphdata.x.size()[0]
    yforce=ytrue[1:].reshape(-1,3)
    yforce_pred=pred_mean[1].squeeze() #.reshape(-1,3)
    yforce_std=pred_std[1].squeeze() #.reshape(-1,3)
    print(yforce.size(), yforce_pred.size())
    for inode in range(num_nodes):
        for icomp in range(3):
         print("Forces, node %d, component, %d True VS Pred at Uncertainty: %.2e, %.2e, %.3e"%(inode, icomp, 
         yforce[inode, icomp].item(), yforce_pred[inode, icomp].item(), yforce_std[inode, icomp].item()))
    ##################################################################################################################
    sys.exit(0)
