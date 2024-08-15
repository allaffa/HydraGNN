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
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
        test_sampler_shuffle=False,
    )
    ##################################################################################################################
    model_ens = model_ensemble(modeldirlist)
    model_ens = hydragnn.utils.get_distributed_model(model_ens, verbosity)
    model_ens.eval()
    ##################################################################################################################
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for icol, (loader, setname) in enumerate(zip([train_loader, val_loader, test_loader], ["train", "val", "test"])):
        error, rmse_task, true_values, predicted_mean, predicted_std = test_ens(model_ens, loader, verbosity, num_samples=200)
        print_distributed(verbosity,"number of heads %d"%len(true_values))
        print_distributed(verbosity,"number of samples %d"%len(true_values[0]))
        if hydragnn.utils.get_comm_size_and_rank()[1]==0:
            print("loss=", error, rmse_task)
        assert len(true_values)==len(predicted_mean), "inconsistent number of heads, %d!=%d"%(len(true_values),len(len(predicted_mean)))
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            head_true = true_values[ihead].cpu().squeeze().numpy() 
            head_pred = predicted_mean[ihead].cpu().squeeze().numpy() 
            head_pred_std = predicted_std[ihead].cpu().squeeze().numpy() 
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]

            ax = axs[ihead, icol]
            error_mae = np.mean(np.abs(head_pred - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
            if hydragnn.utils.get_comm_size_and_rank()[1]==0:
                print(varname, ": mae=", error_mae, ", rmse= ", error_rmse)
            hist2d_norm = getcolordensity(head_true, head_pred)
            ax.errorbar(head_true, head_pred, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            ax.scatter(head_true, head_pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(head_pred), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname, fontsize=24)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2e}".format(error_mae),
            )
    fig.savefig("./logs/" + log_name + "/parity_plot_all.png")
    plt.close()
    ##################################################################################################################
    sys.exit(0)
