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
import hydragnn
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


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
    ##################################################################################################################
    parser = argparse.ArgumentParser()
    print("gfm starting")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--models_dir_folder", help="folder of trained models", type=str, default=None)
    parser.add_argument("--log", help="log name", default="GFM_EnsembleInference")
    parser.add_argument("--multi_model_list", help="multidataset list", default="OC2020")
    args = parser.parse_args()
    ##################################################################################################################
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
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)
    ##################################################################################################################
    ##################################################################################################################
    log_name = "GFM_EnsembleInference" if args.log is None else args.log
    ##################################################################################################################
    modellist = args.multi_model_list.split(",")
    ##################################################################################################################
    def get_ensemble_mean_std(file_name):
        loaded = torch.load(file_name)
        true_values=loaded['true']
        head_pred_ens=loaded['pred_ens']
        x_atomnums=loaded['atomnums']
        graph_batch=loaded['graph_batch']
        print(head_pred_ens.size(), x_atomnums.size(), graph_batch.size())
        print(x_atomnums[:50], graph_batch[:50])
        #print_distributed(verbosity,"number of samples %d"%len(true_values))
        head_pred_mean = head_pred_ens.mean(axis=0)
        head_pred_std = head_pred_ens.std(axis=0)
        head_true = true_values.cpu().squeeze().numpy() 
        head_pred_ens = head_pred_ens.cpu().squeeze().numpy() 
        head_pred_mean = head_pred_mean.cpu().squeeze().numpy() 
        head_pred_std = head_pred_std.cpu().squeeze().numpy() 
        x_atomnums=x_atomnums.cpu().squeeze().numpy()
        graph_batch=graph_batch.cpu().squeeze().numpy()
        return head_true, head_pred_mean, head_pred_std
    ##################################################################################################################
    nheads = len(config["NeuralNetwork"]["Variables_of_interest"]["output_names"])
    """
    for dataset in modellist:
        fig, axs = plt.subplots(nheads, 3, figsize=(18, 6*nheads))
        for icol,  setname in enumerate(["train", "val", "test"]):  
            saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
            for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
            config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
            config["NeuralNetwork"]["Variables_of_interest"]["type"],
            config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
            )): 
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
                ifeat = var_config["output_index"][ihead]
                outtype = var_config["type"][ihead]
                varname = var_config["output_names"][ihead]
                try:
                    ax = axs[ihead, icol]
                except:
                    ax = axs[icol]
                error_mae = np.mean(np.abs(head_pred_mean - head_true))
                error_rmse = np.sqrt(np.mean(np.abs(head_pred_mean - head_true) ** 2))
                if hydragnn.utils.get_comm_size_and_rank()[1]==0:
                    print(setname, varname, ": mae=", error_mae, ", rmse= ", error_rmse)
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
                if ihead==1:
                    ax.set_xlabel("True")
                plt.colorbar(sc)
                ax.set_aspect('equal', adjustable='box')
        fig.savefig("./logs/" + log_name + "/parity_plot_all_errorbar_"+dataset+".png",dpi=500)
        fig.savefig("./logs/" + log_name + "/parity_plot_all_errorbar_"+dataset+".pdf")
        plt.close()
    """
    ##################################################################################################################
    linestyle=["-","--","-.",":","-","--","-.",":"]
    """
    fig, axs = plt.subplots(1, nheads, figsize=(12, 6))
    for icol, setname in enumerate(["test"]):
        for idataset, dataset in enumerate(modellist):
            saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
            for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
            config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
            config["NeuralNetwork"]["Variables_of_interest"]["type"],
            config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
            )):
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
                ifeat = var_config["output_index"][ihead]
                outtype = var_config["type"][ihead]
                varname = var_config["output_names"][ihead]
                ax = axs[ihead]

                #hist1d, bin_edges = np.histogram(head_pred_std, bins=50)
                _, bins = np.histogram(np.log10(head_pred_std), bins=40 )#'auto')
                hist1d, bin_edges = np.histogram(head_pred_std, bins=10**bins)
                ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d/sum(hist1d), linestyle[idataset], linewidth=2.0, label=dataset)
                #ax.set_title(setname + "; " + varname, fontsize=24)
                ax.set_title(varname, fontsize=24)
                if ihead==0:
                    ax.set_ylabel("Ratio")
                ax.set_xlabel("Uncertainty")
                ax.set_xscale('log')
    #axs[0].legend()
    axs[1].legend()
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.925, wspace=0.2, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/uncertainties_testset_hist_plot_"+'-'.join(modellist)+".png")
    fig.savefig("./logs/" + log_name + "/uncertainties_testset_hist_plot_"+'-'.join(modellist)+".pdf")
    plt.close()
    ##################################################################################################################
    fig, axs = plt.subplots(nheads, 3, figsize=(18, 6*nheads))
    for icol, setname in enumerate(["train", "val", "test"]):
        for dataset in modellist:
            saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
            for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
            config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
            config["NeuralNetwork"]["Variables_of_interest"]["type"],
            config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
            )):
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
                ifeat = var_config["output_index"][ihead]
                outtype = var_config["type"][ihead]
                varname = var_config["output_names"][ihead]
                try:
                    ax = axs[ihead, icol]
                except:
                    ax = axs[icol]

                #hist1d, bin_edges = np.histogram(head_pred_std, bins=50)
                _, bins = np.histogram(np.log10(head_pred_std), bins='auto')
                hist1d, bin_edges = np.histogram(head_pred_std, bins=10**bins)

                ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "-", label=dataset)
                ax.set_title(setname + "; " + varname, fontsize=24)
                if icol==0:
                    ax.set_ylabel("Number of points")
                if ihead==1:
                    ax.set_xlabel("Uncertainties")
                ax.set_xscale('log')
    ax.legend()
    fig.savefig("./logs/" + log_name + "/uncertainties_hist_plot_"+'-'.join(modellist)+".png")
    plt.close()
    """
    ##################################################################################################################
    fig, axs = plt.subplots(nheads, 3, figsize=(18, 6*nheads))
    for icol,  setname in enumerate(["train", "val", "test"]):  
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
            config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
            config["NeuralNetwork"]["Variables_of_interest"]["type"],
            config["NeuralNetwork"]["Variables_of_interest"]["output_dim"])): 
            try:
                ax = axs[ihead, icol]
            except:
                ax = axs[icol]
            for dataset in modellist:
                saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
                ifeat = var_config["output_index"][ihead]
                outtype = var_config["type"][ihead]
                varname = var_config["output_names"][ihead]
                error_mae = np.mean(np.abs(head_pred_mean - head_true))
                error_rmse = np.sqrt(np.mean(np.abs(head_pred_mean - head_true) ** 2))
                if hydragnn.utils.get_comm_size_and_rank()[1]==0:
                    print(setname, dataset, varname, ": mae=", error_mae, ", rmse= ", error_rmse)
                hist2d_norm = getcolordensity(head_true, head_pred_mean)
                ax.errorbar(head_true, head_pred_mean, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
                sc=ax.scatter(head_true, head_pred_mean, s=12, c=hist2d_norm, vmin=0, vmax=1)
                minv = np.minimum(np.amin(head_pred_mean), np.amin(head_true))
                maxv = np.maximum(np.amax(head_pred_mean), np.amax(head_true))
                ax.plot([minv, maxv], [minv, maxv], "r--")
                ax.set_title(setname + "; " + varname, fontsize=24)
                #ax.text(
                #    minv + 0.1 * (maxv - minv),
                #    maxv - 0.1 * (maxv - minv),
                #    "MAE: {:.2e}".format(error_mae),
                #)
                if icol==0:
                    ax.set_ylabel("Predicted")
                if ihead==1:
                    ax.set_xlabel("True")
            plt.colorbar(sc)
            ax.set_aspect('equal', adjustable='box')
    for icol in range(3):
        for irow in range(2):
            ax=axs[irow, icol]
            xmin, xmax = ax.get_ylim()
            ymin, ymax = ax.get_ylim()
            ax.set_xlim(min(xmin, ymin), max(xmax,ymax))
            ax.set_ylim(min(xmin, ymin), max(xmax,ymax))
            ax.set_aspect('equal', adjustable='box')

    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.925, wspace=0.2, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parity_plot_all_errorbar_"+'-'.join(modellist)+".png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parity_plot_all_errorbar_"+'-'.join(modellist)+".pdf")
    plt.close()
    ##################################################################################################################
    """
    fig, axs = plt.subplots(nheads, 3, figsize=(18, 6*nheads))
    for icol, setname in enumerate(["train", "val", "test"]):
        saveresultsto=f"./logs/{log_name}/{'-'.join(modellist)}_{setname}_"
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            file_name= saveresultsto +"head%d_atomnum_batch_%s.db"%(ihead, "cuda:0")
            head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            try:
                ax = axs[ihead, icol]
            except:
                ax = axs[icol]
            error_mae = np.mean(np.abs(head_pred_mean - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred_mean - head_true) ** 2))
            if hydragnn.utils.get_comm_size_and_rank()[1]==0:
                print(setname, varname, ": mae=", error_mae, ", rmse= ", error_rmse)
            hist1d, bin_edges = np.histogram(head_pred_mean - head_true, bins=50)
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "-")
            ax.set_title(setname + "; " + varname+" MAE: {:.2e}".format(error_mae), fontsize=24)
            
            if icol==0:
                ax.set_ylabel("Number of points")
            if ihead==1:
                ax.set_xlabel("Error=Pred-True")
            #plt.colorbar(sc)
    fig.savefig("./logs/" + log_name + "/errorhist_plot_"+'-'.join(modellist)+".png")
    plt.close()
    """
    ##################################################################################################################
    sys.exit(0)
   
