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
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({"font.size": 24})


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
    ##################################################################################################################
    linestyle=["-","--","-.",":","-","--","-.",":"]
    fig, axs = plt.subplots(nheads, 1, figsize=(7, 10))
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
                ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d/sum(hist1d), linestyle[idataset], linewidth=3.0, label=dataset)
                #ax.set_title(setname + "; " + varname, fontsize=24)
                ax.set_title(varname, fontsize=28)
                if ihead==1:
                    ax.set_xlabel("Uncertainty $\sigma_{\\tilde y}$", fontsize=28)
                ax.set_ylabel("Count Ratio", fontsize=28)
                ax.set_xscale('log')
    #axs[0].legend()
    axs[1].legend(fontsize=20)
    plt.subplots_adjust(left=0.20, bottom=0.1, right=0.97, top=0.95, wspace=0.2, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/uncertainties_testset_hist_plot_"+'-'.join(modellist)+"_vertical.png")
    fig.savefig("./logs/" + log_name + "/uncertainties_testset_hist_plot_"+'-'.join(modellist)+"_vertical.pdf")
    plt.close()

    fig, axs = plt.subplots(nheads, 1, figsize=(7, 10))
    for icol, setname in enumerate(["test"]):
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            true=[]
            pred=[]
            for idataset, dataset in enumerate(modellist):
                saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
         
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
                true.extend(head_true)
                pred.extend(head_pred_mean)
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            ax = axs[ihead]

            hist2d_norm = getcolordensity(true, pred)
            #ax.errorbar(head_true, head_pred_mean, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(true, pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(pred), np.amin(true))
            maxv = np.maximum(np.amax(pred), np.amax(true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            #ax.set_title(setname + "; " + varname, fontsize=24)
            ax.set_title(varname, fontsize=28)
            if ihead==1:
                ax.set_xlabel("True value", fontsize=28)
            ax.set_ylabel("Predicted value", fontsize=28) # $\\tilde y$
            cbar=plt.colorbar(sc)
            cbar.ax.set_ylabel('Density', rotation=90)
            ax.set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.95, wspace=0.175, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parityplot_testset_scatter_plot_"+'-'.join(modellist)+"_vertical.png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parityplot_testset_scatter_plot_"+'-'.join(modellist)+"_vertical.pdf")
    plt.close()

    fig, axs = plt.subplots(1, nheads, figsize=(12, 5))
    for icol, setname in enumerate(["test"]):
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            true=[]
            pred=[]
            for idataset, dataset in enumerate(modellist):
                saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
         
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
                true.extend(head_true)
                pred.extend(head_pred_mean)
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            ax = axs[ihead]

            hist2d_norm = getcolordensity(true, pred)
            #ax.errorbar(head_true, head_pred_mean, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(true, pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(pred), np.amin(true))
            maxv = np.maximum(np.amax(pred), np.amax(true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            #ax.set_title(setname + "; " + varname, fontsize=24)
            ax.set_title(varname, fontsize=28)
            
            ax.set_xlabel("True value", fontsize=28)
            if ihead==0:
                ax.set_ylabel("Predicted value", fontsize=28) # $\\tilde y$
            #cbar=plt.colorbar(sc)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if ihead==0:
                cax.set_axis_off()
            else:
                cbar=fig.colorbar(sc, cax=cax, orientation='vertical')
                cbar.ax.set_ylabel('Density', rotation=90)
            ax.set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0.1, bottom=0.175, right=0.9, top=0.91, wspace=0.15, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parityplot_testset_scatter_plot_"+'-'.join(modellist)+"_horizontal.png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parityplot_testset_scatter_plot_"+'-'.join(modellist)+"_horizontal.pdf")
    plt.close()
    
    fig, axs = plt.subplots(nheads, 1, figsize=(7, 10))
    for icol, setname in enumerate(["test"]):
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            true=[]
            pred=[]
            for idataset, dataset in enumerate(modellist):
                saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
         
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
                true.extend(head_true)
                pred.extend(head_pred_mean)
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            ax = axs[ihead]

            hist2d_norm = getcolordensity(true, pred)
            ax.errorbar(head_true, head_pred_mean, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(true, pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(pred), np.amin(true))
            maxv = np.maximum(np.amax(pred), np.amax(true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            #ax.set_title(setname + "; " + varname, fontsize=24)
            ax.set_title(varname, fontsize=28)
            if ihead==1:
                ax.set_xlabel("True value", fontsize=28)
            ax.set_ylabel("Predicted value", fontsize=28) # $\\tilde y$
            cbar=plt.colorbar(sc)
            cbar.ax.set_ylabel('Density', rotation=90)
            ax.set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.95, wspace=0.175, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parityplot_testset_errorbar_plot_"+'-'.join(modellist)+"_vertical.png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parityplot_testset_errorbar_plot_"+'-'.join(modellist)+"_vertical.pdf")
    plt.close()

    fig, axs = plt.subplots(nheads, 2, figsize=(12, 10))
    for icol, setname in enumerate(["test"]):
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            true=[]
            pred=[]
            for idataset, dataset in enumerate(modellist):
                saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
         
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
                true.extend(head_true)
                pred.extend(head_pred_mean)
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            ax = axs[ihead, 0]

            hist2d_norm = getcolordensity(true, pred)
            #ax.errorbar(head_true, head_pred_mean, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(true, pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(pred), np.amin(true))
            maxv = np.maximum(np.amax(pred), np.amax(true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            #ax.set_title(setname + "; " + varname, fontsize=24)
            ax.set_title(varname, fontsize=28)
            if ihead==1:
                ax.set_xlabel("True value", fontsize=28)
            ax.set_ylabel("Predicted value", fontsize=28) # $\\tilde y$
            #cbar=plt.colorbar(sc)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cax.set_axis_off()
            #cbar.ax.set_ylabel('Density', rotation=90)
            ax.set_aspect('equal', adjustable='box')
    
    for icol, setname in enumerate(["test"]):
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            true=[]
            pred=[]
            for idataset, dataset in enumerate(modellist):
                saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
         
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
                true.extend(head_true)
                pred.extend(head_pred_mean)
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            ax = axs[ihead, 1]

            hist2d_norm = getcolordensity(true, pred)
            ax.errorbar(head_true, head_pred_mean, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(true, pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(pred), np.amin(true))
            maxv = np.maximum(np.amax(pred), np.amax(true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            #ax.set_title(setname + "; " + varname, fontsize=24)
            ax.set_title(varname, fontsize=28)
            if ihead==1:
                ax.set_xlabel("True value", fontsize=28)
            ax.set_ylabel("Predicted value", fontsize=28) # $\\tilde y$
            #cbar=plt.colorbar(sc)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar=fig.colorbar(sc, cax=cax, orientation='vertical')
            #cbar.ax.set_ylabel('Density', rotation=90)
            cbar.ax.set_ylabel('Density', rotation=90)
            ax.set_aspect('equal', adjustable='box')
            xmin, xmax = ax.get_ylim()
            ymin, ymax = ax.get_ylim()
            ax.set_xlim(min(xmin, ymin), max(xmax,ymax))
            ax.set_ylim(min(xmin, ymin), max(xmax,ymax))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.925, top=0.95, wspace=0.15, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parityplot_testset_scatter_errorbar_plot_"+'-'.join(modellist)+"_vertical.png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parityplot_testset_scatter_errorbar_plot_"+'-'.join(modellist)+"_vertical.pdf")
    plt.close()
    ##################################################################################################################
    linestyle=["-","--","-.",":","-","--","-.",":"]
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
                ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d/sum(hist1d), linestyle[idataset], linewidth=3.0, label=dataset)
                #ax.set_title(setname + "; " + varname, fontsize=24)
                ax.set_title(varname, fontsize=28)
                if ihead==0:
                    ax.set_ylabel("Count Ratio", fontsize=28)
                ax.set_xlabel("Uncertainty $\sigma_{\\tilde y}$", fontsize=28)
                ax.set_xscale('log')
    #axs[0].legend()
    axs[1].legend(fontsize=20)
    plt.subplots_adjust(left=0.12, bottom=0.16, right=0.99, top=0.925, wspace=0.2, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/uncertainties_testset_hist_plot_"+'-'.join(modellist)+".png")
    fig.savefig("./logs/" + log_name + "/uncertainties_testset_hist_plot_"+'-'.join(modellist)+".pdf")
    plt.close()

    fig, axs = plt.subplots(1, nheads, figsize=(12, 5))
    for icol, setname in enumerate(["test"]):
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            true=[]
            pred=[]
            for idataset, dataset in enumerate(modellist):
                saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
         
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_mean, head_pred_std = get_ensemble_mean_std(file_name)
                true.extend(head_true)
                pred.extend(head_pred_mean)
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            ax = axs[ihead]

            hist2d_norm = getcolordensity(true, pred)
            #ax.errorbar(head_true, head_pred_mean, yerr=head_pred_std, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(true, pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(pred), np.amin(true))
            maxv = np.maximum(np.amax(pred), np.amax(true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            #ax.set_title(setname + "; " + varname, fontsize=24)
            ax.set_title(varname, fontsize=28)
            if ihead==0:
                ax.set_ylabel("Predicted value", fontsize=28) # $\\tilde y$
            ax.set_xlabel("True value", fontsize=28)
            cbar=plt.colorbar(sc)
            cbar.ax.set_ylabel('Density', rotation=90)
            ax.set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.925, wspace=0.175, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parityplot_testset_scatter_plot_"+'-'.join(modellist)+".png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parityplot_testset_scatter_plot_"+'-'.join(modellist)+".pdf")
    plt.close()

    fig, axs = plt.subplots(nheads, 2, figsize=(12, 12))
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
                ax = axs[ihead,0]

                #hist1d, bin_edges = np.histogram(head_pred_std, bins=50)
                _, bins = np.histogram(np.log10(head_pred_std), bins=40 )#'auto')
                hist1d, bin_edges = np.histogram(head_pred_std, bins=10**bins)
                ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d/sum(hist1d), linestyle[idataset], linewidth=3.0, label=dataset)
                #ax.set_title(setname + "; " + varname, fontsize=24)
                ax.set_title(varname, fontsize=28)
                #if ihead==0:
                ax.set_ylabel("Count Ratio", fontsize=28)
                ax.set_xlabel("Uncertainty $\sigma_{\\tilde y}$", fontsize=28)
                ax.set_xscale('log')
    axs[1,0].legend(fontsize=20)
    data_stat={"energy":{
            "ANI1x-v3":[-0.03907776, 0.10897827, 5.614973e-08, 0.0064779613],
            "MPTrj-v3":[-4.3795958, 9.891445, -7.966261e-10, 0.6926824],
            "qm7x-v3":[ -0.5088501, 1.1589355, 2.8013553e-06, 0.16951331],
            "OC2022-v3":[-277.11557, 140.97452, -1.0578678e-09, 0.42587683],
            "OC2020-v3":[-129.0625, 10.31657, -4.0010367e-10, 0.26403916],
            "OC2020-20M-v3":[-30.808363, 10.279333, 3.4940048e-10, 0.2637729],
        },
        "forces":{
            "ANI1x-v3":[-3.1214485, 3.1448452, 1.2772762e-12, 0.0782825],
            "MPTrj-v3":[-95.6116, 95.6116, -1.0003237e-11, 0.72286475],
            "qm7x-v3":[-51.879436, 53.1437, -2.2875093e-07, 1.6194862],
            "OC2022-v3":[ -99.540054, 99.53778, -5.093522e-14, 0.37664887],
            "OC2020-v3":[ -49.99948, 49.998825, -1.8164947e-13, 0.43650356],
            "OC2020-20M-v3":[-49.994606, 49.99851, 1.603377e-13, 0.43715373],
        },
    }
    """
    #min, max, mean, and std
    ANI1x-v2 -0.03907776 0.10897827 5.614973e-08 0.0064779613
    MPTrj-v2 -4.3795958 9.891445 -7.966261e-10 0.6926824
    qm7x-v2 -0.5088501 1.1589355 2.8013553e-06 0.16951331
    OC2022-v2 -277.11557 140.97452 -1.0578678e-09 0.42587683
    OC2020-v2 -129.0625 10.31657 -4.0010367e-10 0.26403916
    OC2020-20M-v2 -30.808363 10.279333 3.4940048e-10 0.2637729

    ANI1x-v2 -3.1214485 3.1448452 1.2772762e-12 0.0782825
    ANI1x-v2 -2.791656 2.8167198 -3.754489e-13 0.0783702
    ANI1x-v2 -2.9283044 2.7735653 -1.9210675e-12 0.07837329

    MPTrj-v2 -95.6116 95.6116 -1.0003237e-11 0.72286475
    MPTrj-v2 -88.0109 86.37443 -2.2971236e-12 0.75364137
    MPTrj-v2 -78.65471 78.65471 -7.822072e-12 0.68438387
    
    qm7x-v2 -51.879436 53.1437 -2.2875093e-07 1.6194862
    qm7x-v2 -38.9852 40.148083 -2.6717044e-07 1.6206008
    qm7x-v2 -31.404245 32.363125 -2.6225158e-07 1.6192709

    OC2022-v2 -99.540054 99.53778 -5.093522e-14 0.37664887
    OC2022-v2 -95.81782 97.73413 2.9950353e-14 0.37322345
    OC2022-v2 -97.29042 97.38194 7.594088e-12 0.41075358

    OC2020-v2 -49.99948 49.998825 -1.8164947e-13 0.43650356
    OC2020-v2 -49.992107 49.99851 5.297463e-13 0.4366304
    OC2020-v2 -49.92771 49.99126 3.915245e-12 0.42873722

    OC2020-20M-v2 -49.994606 49.99851 1.603377e-13 0.43715373
    OC2020-20M-v2 -49.999195 49.93051 -1.1316512e-12 0.4359669
    OC2020-20M-v2 -49.92771 49.99126 3.915245e-12 0.42873722
    """
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
                ax = axs[ihead,1]
                vmin, vmax, vmean, vstd = data_stat[varname][dataset]
                scale_uq=1.0/vstd

                #hist1d, bin_edges = np.histogram(head_pred_std, bins=50)
                _, bins = np.histogram(np.log10(head_pred_std), bins=40 )#'auto')
                hist1d, bin_edges = np.histogram(head_pred_std, bins=10**bins)
                ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:])*scale_uq, hist1d/sum(hist1d), linestyle[idataset], linewidth=3.0, label=dataset)
                #ax.set_title(setname + "; " + varname, fontsize=24)
                ax.set_title(varname, fontsize=28)
                #if ihead==0:
                #    ax.set_ylabel("Count Ratio", fontsize=28)
                ax.set_xlabel("Relative Uncertainty $\sigma_{\\tilde y}/\sigma_D$", fontsize=28)
                ax.set_xscale('log')
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.98, top=0.95, wspace=0.2, hspace=0.35)
    fig.savefig("./logs/" + log_name + "/uncertainties_two_testset_hist_plot_"+'-'.join(modellist)+".png")
    fig.savefig("./logs/" + log_name + "/uncertainties_two_testset_hist_plot_"+'-'.join(modellist)+".pdf")
    plt.close()
    ##################################################################################################################
    sys.exit(0)
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

    plt.subplots_adjust(left=0.075, bottom=0.1, right=0.975, top=0.95, wspace=0.2, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parity_plot_all_errorbar_"+'-'.join(modellist)+".png",dpi=500)
    fig.savefig("./logs/" + log_name + "/parity_plot_all_errorbar_"+'-'.join(modellist)+".pdf")
    plt.close()
    ##################################################################################################################
    sys.exit(0)
   
