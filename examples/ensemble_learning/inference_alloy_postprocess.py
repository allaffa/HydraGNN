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
import argparse

import torch
import numpy as np

#from hydragnn.utils.print_utils import print_distributed
from scipy.interpolate import griddata


import matplotlib.pyplot as plt
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
    parser.add_argument("--models_dir_folder", help="folder of trained models", type=str, default="./examples/ensemble_learning/alloy_binary_rmsd_1,examples/ensemble_learning/alloy_binary_rmsd_2")
    parser.add_argument("--log", help="log name", default="EL_rmsd2") 
    parser.add_argument("--nprocs", help="number of GPUs used in UQ", type=int, default=16) 
    #parser.add_argument("--models_dir_folder", help="folder of trained models", type=str, default="./examples/ensemble_learning/alloy_binary_energy")
    #parser.add_argument("--log", help="log name", default="EL_energy2") 
    #parser.add_argument("--models_dir_folder", help="folder of trained models", type=str, default="./examples/ensemble_learning/alloy_binary_lattice")
    #parser.add_argument("--log", help="log name", default="EL_lattice2") 
    args = parser.parse_args()


# python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_rmsd_1,examples/ensemble_learning/alloy_binary_rmsd_2  --dataname=alloy_binary_rmsd --log="EL_rmsd2"
# python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_energy --log="EL_energy2"
# python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_lattice --dataname=alloy_binary_lattice --log="EL_lattice2"

    nprocs = args.nprocs
    modeldirlists = args.models_dir_folder.split(",")
    assert len(modeldirlists)==1 or len(modeldirlists)==2
    if len(modeldirlists)==1:
        modeldirlist = [os.path.join(args.models_dir_folder, name) for name in os.listdir(args.models_dir_folder) if os.path.isdir(os.path.join(args.models_dir_folder, name))]
    else:
        modeldirlist = []
        for models_dir_folder in modeldirlists:
            modeldirlist.extend([os.path.join(models_dir_folder, name) for name in os.listdir(models_dir_folder) if os.path.isdir(os.path.join(models_dir_folder, name))])


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
    ##################################################################################################################
    def get_ensemble_mean_std(file_name):
        loaded = torch.load(file_name)
        true_values=loaded['true']
        head_pred_ens=loaded['pred_ens']
        print(file_name, head_pred_ens.size(), true_values.size())
        #print_distributed(verbosity,"number of samples %d"%len(true_values))
        head_pred_mean = head_pred_ens.mean(axis=0)
        head_pred_std = head_pred_ens.std(axis=0)
        head_true = true_values.cpu().squeeze().numpy() 
        head_pred_ens = head_pred_ens.cpu().squeeze().numpy() 
        head_pred_mean = head_pred_mean.cpu().squeeze().numpy() 
        head_pred_std = head_pred_std.cpu().squeeze().numpy() 
        return head_true, head_pred_mean, head_pred_std
    ##################################################################################################################
    ##################################################################################################################
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for icol, setname in enumerate(["train", "val", "test"]):
        saveresultsto=f"./logs/{log_name}/{setname}_"

        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            assert ihead==0
            ax = axs[icol]
            """
            file_name= saveresultsto +"head%d.db"%ihead
            loaded = torch.load(file_name)
            true_values=loaded['true']
            head_pred_ens=loaded['pred_ens']
            print(head_pred_ens.size())
            #print_distributed(verbosity,"number of samples %d"%len(true_values))
            head_pred_mean = head_pred_ens.mean(axis=0)
            head_pred_std = head_pred_ens.std(axis=0)
            head_true = true_values.cpu().squeeze().numpy() 
            head_pred_ens = head_pred_ens.cpu().squeeze().numpy() 
            head_pred_mean = head_pred_mean.cpu().squeeze().numpy() 
            head_pred_std = head_pred_std.cpu().squeeze().numpy() 
            """ 
            head_true=[None]*nprocs
            head_pred_mean=[None]*nprocs
            head_pred_std=[None]*nprocs
            for iproc in range(nprocs):    
                file_name= saveresultsto +"head%d_proc%d.db"%(ihead, iproc)
                head_true[iproc], head_pred_mean[iproc], head_pred_std[iproc] = get_ensemble_mean_std(file_name)
            head_true=np.concatenate(head_true)
            head_pred_mean=np.concatenate(head_pred_mean)
            head_pred_std=np.concatenate(head_pred_std)

            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            

            error_mae = np.mean(np.abs(head_pred_mean - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred_mean - head_true) ** 2))
            
            hist2d_norm = getcolordensity(head_true, head_pred_mean)
            #ax.errorbar(head_true, head_pred, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(head_true, head_pred_mean, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(head_pred_mean), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred_mean), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname, fontsize=24)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2e}".format(error_mae),
            )
            if icol==0:
                ax.set_ylabel("Predicted")
            ax.set_xlabel("True")
            ax.set_aspect('equal', adjustable='box')
            #plt.colorbar(sc)
            if True: #icol==2:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                if icol==2:
                    fig.colorbar(sc, cax=cax, orientation='vertical')
                else:
                    cax.set_axis_off()
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parity_plot_post.png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parity_plot_post.pdf")
    plt.close()
    ##################################################################################################################
    ##################################################################################################################
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for icol, setname in enumerate(["train", "val", "test"]):
        saveresultsto=f"./logs/{log_name}/{setname}_"

        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            assert ihead==0
            ax = axs[icol]
            """
            file_name= saveresultsto +"head%d.db"%ihead
            loaded = torch.load(file_name)
            true_values=loaded['true']
            head_pred_ens=loaded['pred_ens']
            print(head_pred_ens.size())
            #print_distributed(verbosity,"number of samples %d"%len(true_values))
            head_pred_mean = head_pred_ens.mean(axis=0)
            head_pred_std = head_pred_ens.std(axis=0)
            head_true = true_values.cpu().squeeze().numpy() 
            head_pred_ens = head_pred_ens.cpu().squeeze().numpy() 
            head_pred_mean = head_pred_mean.cpu().squeeze().numpy() 
            head_pred_std = head_pred_std.cpu().squeeze().numpy() 
            """ 
            head_true=[None]*nprocs
            head_pred_mean=[None]*nprocs
            head_pred_std=[None]*nprocs
            for iproc in range(nprocs):    
                file_name= saveresultsto +"head%d_proc%d.db"%(ihead, iproc)
                head_true[iproc], head_pred_mean[iproc], head_pred_std[iproc] = get_ensemble_mean_std(file_name)
            head_true=np.concatenate(head_true)
            head_pred_mean=np.concatenate(head_pred_mean)
            head_pred_std=np.concatenate(head_pred_std)

            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            

            error_mae = np.mean(np.abs(head_pred_mean - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred_mean - head_true) ** 2))
            
            hist2d_norm = getcolordensity(head_true, head_pred_mean)
            ax.errorbar(head_true, head_pred_mean, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(head_true, head_pred_mean, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(head_pred_mean), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred_mean), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname, fontsize=24)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2e}".format(error_mae),
            )
            if icol==0:
                ax.set_ylabel("Predicted")
            ax.set_xlabel("True")
            ax.set_aspect('equal', adjustable='box')
            #plt.colorbar(sc)
            if True: #icol==2:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                if icol==2:
                    fig.colorbar(sc, cax=cax, orientation='vertical')
                else:
                    cax.set_axis_off()
            xmin, xmax = ax.get_ylim()
            ymin, ymax = ax.get_ylim()
            ax.set_xlim(min(xmin, ymin), max(xmax,ymax))
            ax.set_ylim(min(xmin, ymin), max(xmax,ymax))
            ax.set_aspect('equal', adjustable='box')
            
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parity_plot_post_errorbar.png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parity_plot_post_errorbar.pdf")
    plt.close()
    ##################################################################################################################
    ##################################################################################################################
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for icol, setname in enumerate(["train", "val", "test"]):
        saveresultsto=f"./logs/{log_name}/{setname}_"

        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            assert ihead==0
            head_true=[None]*nprocs
            head_pred_mean=[None]*nprocs
            head_pred_std=[None]*nprocs
            for iproc in range(nprocs):    
                file_name= saveresultsto +"head%d_proc%d.db"%(ihead, iproc)
                head_true[iproc], head_pred_mean[iproc], head_pred_std[iproc] = get_ensemble_mean_std(file_name)
            head_true=np.concatenate(head_true)
            head_pred_mean=np.concatenate(head_pred_mean)
            head_pred_std=np.concatenate(head_pred_std)

            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            
            hist1d, bin_edges = np.histogram(head_pred_std, bins=50)
            #_, bins = np.histogram(np.log10(head_pred_std), bins='auto')
            #hist1d, bin_edges = np.histogram(head_pred_std, bins=10**bins)

            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d/sum(hist1d), "-", label=setname+"; "+str(sum(hist1d)))
        
        ax.set_title(varname, fontsize=24)
        ax.set_ylabel("Count Ratio", fontsize=28)
        ax.set_xlabel("Uncertainties")
         
    ax.legend()
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.98, top=0.925)#, wspace=0.2, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/hist_uncertainty.png",dpi=500)
    fig.savefig("./logs/" + log_name + "/hist_uncertainty.pdf")
    plt.close()
    ##################################################################################################################
    """
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
            #print_distributed(verbosity,"number of samples %d"%len(true_values))
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
    """
    ##################################################################################################################
    sys.exit(0)
