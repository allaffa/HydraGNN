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
from ensemble_utils import model_ensemble, test_ens, debug_nan
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir_folder", help="folder of trained models", type=str, default=None)
    parser.add_argument("--dataname", help="name of dataset pickle file", type=str, default="alloy_binary_energy")
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

    modeldirlists = args.models_dir_folder.split(",")
    assert len(modeldirlists)==1 or len(modeldirlists)==2
    if len(modeldirlists)==1:
        modeldirlist = [os.path.join(args.models_dir_folder, name) for name in os.listdir(args.models_dir_folder) if os.path.isdir(os.path.join(args.models_dir_folder, name))]
    else:
        modeldirlist = []
        for models_dir_folder in modeldirlists:
            modeldirlist.extend([os.path.join(models_dir_folder, name) for name in os.listdir(models_dir_folder) if os.path.isdir(os.path.join(models_dir_folder, name))])

    modelname=args.dataname

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
        8,
        #config["NeuralNetwork"]["Training"]["batch_size"],
        test_sampler_shuffle=False,
    )
    print(hydragnn.utils.get_comm_size_and_rank(), config["NeuralNetwork"]["Training"]["batch_size"], len(train_loader), len(val_loader), len(test_loader))
    ##################################################################################################################
    model_ens = model_ensemble(modeldirlist)
    model_ens = hydragnn.utils.get_distributed_model(model_ens, verbosity)
    model_ens.eval()
    ##################################################################################################################
    nheads = len(config["NeuralNetwork"]["Variables_of_interest"]["output_names"])
    fig, axs = plt.subplots(nheads, 3, figsize=(18, 6*nheads))
    for icol, (loader, setname) in enumerate(zip([train_loader, val_loader, test_loader], ["train", "val", "test"])):
        error, rmse_task, true_values, predicted_mean, predicted_std = test_ens(model_ens, loader, verbosity) #, num_samples=1024)
        print_distributed(verbosity,"number of heads %d"%len(true_values))
        print_distributed(verbosity,"number of samples %d"%len(true_values[0]))
        if hydragnn.utils.get_comm_size_and_rank()[1]==0:
            print(setname, "loss=", error, rmse_task)
        assert len(true_values)==len(predicted_mean), "inconsistent number of heads, %d!=%d"%(len(true_values),len(len(predicted_mean)))
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )): 
            _ = debug_nan(true_values[ihead], message="checking on true for %s"%output_name)
            _ = debug_nan(predicted_mean[ihead], message="checking on predicted mean for %s"%output_name)
            head_true = true_values[ihead].cpu().squeeze().numpy() 
            head_pred = predicted_mean[ihead].cpu().squeeze().numpy() 
            head_pred_std = predicted_std[ihead].cpu().squeeze().numpy()
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]

            try:
                ax = axs[ihead, icol]
            except:
                ax = axs[icol]
            error_mae = np.mean(np.abs(head_pred - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
            if hydragnn.utils.get_comm_size_and_rank()[1]==0:
                print(setname, varname, ": mae=", error_mae, ", rmse= ", error_rmse)
            print(hydragnn.utils.get_comm_size_and_rank()[1], head_true.size, head_pred.size)
            hist2d_norm = getcolordensity(head_true, head_pred)
            #ax.errorbar(head_true, head_pred, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(head_true, head_pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(head_pred), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname, fontsize=24)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2e}".format(error_mae),
            )
            if icol==0:
                ax.set_ylabel("Predicted")
            if ihead==1:
                ax.set_xlabel("True")
            #plt.colorbar(sc)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(sc, cax=cax, orientation='vertical')
            #cbar=plt.colorbar(sc)
            #cbar.ax.set_ylabel('Density', rotation=90)
            #ax.set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.4, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parity_plot_all.png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parity_plot_all.pdf")
    plt.close()
    
    fig, axs = plt.subplots(nheads, 3, figsize=(18, 6*nheads))
    for icol, (loader, setname) in enumerate(zip([train_loader, val_loader, test_loader], ["train", "val", "test"])):
        error, rmse_task, true_values, predicted_mean, predicted_std = test_ens(model_ens, loader, verbosity)#, num_samples=1024)
        print_distributed(verbosity,"number of heads %d"%len(true_values))
        print_distributed(verbosity,"number of samples %d"%len(true_values[0]))
        if hydragnn.utils.get_comm_size_and_rank()[1]==0:
            print(setname, "loss=", error, rmse_task)
        assert len(true_values)==len(predicted_mean), "inconsistent number of heads, %d!=%d"%(len(true_values),len(len(predicted_mean)))
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            _ = debug_nan(true_values[ihead], message="checking on true for %s"%output_name)
            _ = debug_nan(predicted_mean[ihead], message="checking on predicted mean for %s"%output_name)
            head_true = true_values[ihead].cpu().squeeze().numpy() 
            head_pred = predicted_mean[ihead].cpu().squeeze().numpy() 
            head_pred_std = predicted_std[ihead].cpu().squeeze().numpy() 
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]

            try:
                ax = axs[ihead, icol]
            except:
                ax = axs[icol]
            error_mae = np.mean(np.abs(head_pred - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
            if hydragnn.utils.get_comm_size_and_rank()[1]==0:
                print(setname, varname, ": mae=", error_mae, ", rmse= ", error_rmse)
            hist2d_norm = getcolordensity(head_true, head_pred)
            ax.errorbar(head_true, head_pred, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(head_true, head_pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(head_pred), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname, fontsize=24)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2e}".format(error_mae),
            )
            if icol==0:
                ax.set_ylabel("Predicted")
            if ihead==1:
                ax.set_xlabel("True")
            #cbar=plt.colorbar(sc)
            #cbar.ax.set_ylabel('Density', rotation=90)
            ax.set_aspect('equal', adjustable='box')
    fig.savefig("./logs/" + log_name + "/parity_plot_all_errorbar.png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parity_plot_all_errorbar.pdf")
    plt.close()
    ##################################################################################################################
    fig, axs = plt.subplots(nheads, 3, figsize=(18, 6*nheads))
    for icol, (loader, setname) in enumerate(zip([train_loader, val_loader, test_loader], ["train", "val", "test"])):
        #error, rmse_task, true_values, predicted_mean, predicted_std = test_ens(model_ens, loader, verbosity, num_samples=1000, saveresultsto=f"./logs/{log_name}/{setname}_")
        #error, rmse_task, true_values, predicted_mean, predicted_std = test_ens(model_ens, loader, verbosity,  num_samples=4096, saveresultsto=f"./logs/{log_name}/{setname}_")
        #error, rmse_task, true_values, predicted_mean, predicted_std = test_ens(model_ens, loader, verbosity,  num_samples=8192, saveresultsto=f"./logs/{log_name}/{setname}_")
        error, rmse_task, true_values, predicted_mean, predicted_std = test_ens(model_ens, loader, verbosity, saveresultsto=f"./logs/{log_name}/{setname}_")
        print_distributed(verbosity,"number of heads %d"%len(true_values))
        print_distributed(verbosity,"number of samples %d"%len(true_values[0]))
        if hydragnn.utils.get_comm_size_and_rank()[1]==0:
            print(setname, "loss=", error, rmse_task)
        assert len(true_values)==len(predicted_mean), "inconsistent number of heads, %d!=%d"%(len(true_values),len(len(predicted_mean)))
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            _ = debug_nan(true_values[ihead], message="checking on true for %s"%output_name)
            _ = debug_nan(predicted_mean[ihead], message="checking on predicted mean for %s"%output_name)
            head_true = true_values[ihead].cpu().squeeze().numpy() 
            head_pred = predicted_mean[ihead].cpu().squeeze().numpy() 
            head_pred_std = predicted_std[ihead].cpu().squeeze().numpy() 
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]

            np.savez("./logs/" + log_name + "/"+setname+varname+".npz", head_true, head_pred, head_pred_std)
            try:
                ax = axs[ihead, icol]
            except:
                ax = axs[icol]
            error_mae = np.mean(np.abs(head_pred - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
            if hydragnn.utils.get_comm_size_and_rank()[1]==0:
                print(setname, varname, ": mae=", error_mae, ", rmse= ", error_rmse)
            hist1d, bin_edges = np.histogram(head_pred - head_true, bins=50)
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "-")
            ax.set_title(setname + "; " + varname+" MAE: {:.2e}".format(error_mae), fontsize=24)
            
            if icol==0:
                ax.set_ylabel("Number of points")
            if ihead==1:
                ax.set_xlabel("Error=Pred-True")
    fig.savefig("./logs/" + log_name + "/errorhist_plot_all.png")
    plt.close()
    ##################################################################################################################
    sys.exit(0)
