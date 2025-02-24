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
    ##################################################################################################################
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for icol, setname in enumerate(["train", "val", "test"]):
        saveresultsto=f"./logs/{log_name}/{setname}_"

        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            ax = axs[ihead, icol]

            file_name= saveresultsto +"head%d.db"%ihead
            loaded = torch.load(file_name)
            true_values=loaded['true']
            head_pred_ens=loaded['pred_ens']
            print(head_pred_ens.size())
            print_distributed(verbosity,"number of samples %d"%len(true_values))
            head_pred_mean = head_pred_ens.mean(axis=0)
            head_pred_std = head_pred_ens.std(axis=0)
            head_true = true_values.cpu().squeeze().numpy() 
            head_pred_ens = head_pred_ens.cpu().squeeze().numpy() 
            head_pred_mean = head_pred_mean.cpu().squeeze().numpy() 
            head_pred_std = head_pred_std.cpu().squeeze().numpy() 

            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]

            for imodel in range(head_pred_ens.shape[0]):
                head_pred = head_pred_ens[imodel,:].squeeze()
                hist1d, bin_edges = np.histogram(head_pred - head_true, bins=50)
                ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "-")
                ax.set_title(setname + "; " + varname, fontsize=24)

            
         
            hist1d, bin_edges = np.histogram(head_pred_mean - head_true, bins=50)
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro-")
            ax.plot([0.0, 0.0],[0.0, max(hist1d)*1.2],"k:")
            ax.set_title(setname + "; " + varname, fontsize=24)
            if icol==0:
                ax.set_ylabel("Number of points")
            if ihead==1:
                ax.set_xlabel("Error=Pred-True")
                ax.set_xlim(-0.75, .75)
            else:
                ax.set_xlim(-0.01, 0.01)
    #fig.savefig("./logs/" + log_name + "/errorhist_plot_allmodels.png")
    fig.savefig("./logs/" + log_name + "/errorhist_plot_allmodels_zoomin.png")
    plt.close()
    ##################################################################################################################
    sys.exit(0)
