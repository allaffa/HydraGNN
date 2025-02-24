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
    modeldirlist_real=[]
    for modeldir in modeldirlist:
        input_filename = os.path.join(modeldir, "config.json")
        if not os.path.exists(input_filename):
            continue
        with open(input_filename, "r") as f:
            config = json.load(f)
        if var_config is not None:
            assert var_config==config["NeuralNetwork"]["Variables_of_interest"], "Inconsistent variable config in %s"%input_filename
        else:
            var_config = config["NeuralNetwork"]["Variables_of_interest"]
        modeldirlist_real.append(modeldir)
    
    modeldirlist = modeldirlist_real
    verbosity=config["Verbosity"]["level"]
    log_name = "GFM_EnsembleInference" if args.log is None else args.log
    ##################################################################################################################
    ##################################################################################################################
    def get_ensemble_mean_std(file_name):
        #m = {'true': true_values[ihead], 'pred_ens': head_pred_ens, 'compositions': compositions}
        loaded = torch.load(file_name)
        true_values=loaded['true']
        head_pred_ens=loaded['pred_ens']
        comp_values=loaded['compositions']
        print(file_name, head_pred_ens.size(), true_values.size())
        #print_distributed(verbosity,"number of samples %d"%len(true_values))
        head_pred_mean = head_pred_ens.mean(axis=0)
        head_pred_std = head_pred_ens.std(axis=0)
        head_true = true_values.cpu().squeeze().numpy() 
        head_pred_ens = head_pred_ens.cpu().squeeze().numpy() 
        head_pred_mean = head_pred_mean.cpu().squeeze().numpy() 
        head_pred_std = head_pred_std.cpu().squeeze().numpy() 
        comp_values = comp_values.cpu().squeeze().numpy() 
        return head_true, head_pred_mean, head_pred_std, comp_values
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
            comp_values = [None]*nprocs
            for iproc in range(nprocs):    
                file_name= saveresultsto +"head%d_proc%d.db"%(ihead, iproc)
                head_true[iproc], head_pred_mean[iproc], head_pred_std[iproc], comp_values[iproc] = get_ensemble_mean_std(file_name)
            head_true=np.concatenate(head_true)
            head_pred_mean=np.concatenate(head_pred_mean)
            head_pred_std=np.concatenate(head_pred_std)
            comp_values=np.concatenate(comp_values)

            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            
            if varname=="formation_enthalpy":
                print(f"Uncertainty checking, {varname} outlier values:", head_true[np.where(head_true>=75)])
                print(f"Uncertainty checking, {varname} outlier values with composition:", comp_values[np.where(head_true>=75)])
                head_pred_mean=head_pred_mean[np.where(head_true<75)]
                head_pred_std=head_pred_std[np.where(head_true<75)]
                comp_values=comp_values[np.where(head_true<75)]
                head_true=head_true[np.where(head_true<75)]
            elif varname=="rmsd":
                print(f"Uncertainty checking, {varname} outlier values:", head_true[np.where(head_true<=1e-4)])
                print(f"Uncertainty checking, {varname} outlier values with composition:", comp_values[np.where(head_true<=1e-4)])
                head_pred_mean=head_pred_mean[np.where(head_true>1e-4)]
                head_pred_std=head_pred_std[np.where(head_true>1e-4)]
                comp_values=comp_values[np.where(head_true>1e-4)]
                head_true=head_true[np.where(head_true>1e-4)]

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
    #fig.savefig("./logs/" + log_name + f"/parity_plot_post_{log_name}.png",dpi=500)
    #fig.savefig("./logs/" + log_name + f"/parity_plot_post_{log_name}.pdf")
    fig.savefig("./logs/" + log_name + f"/parity_plot_post_{log_name}_.png",dpi=500)
    fig.savefig("./logs/" + log_name + f"/parity_plot_post_{log_name}_.pdf")
    plt.close()
    ##################################################################################################################
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for irow, setname in enumerate(["train", "val", "test"]):
        saveresultsto=f"./logs/{log_name}/{setname}_"

        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            assert ihead==0
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
            comp_values = [None]*nprocs
            for iproc in range(nprocs):    
                file_name= saveresultsto +"head%d_proc%d.db"%(ihead, iproc)
                head_true[iproc], head_pred_mean[iproc], head_pred_std[iproc], comp_values[iproc] = get_ensemble_mean_std(file_name)
            head_true=np.concatenate(head_true)
            head_pred_mean=np.concatenate(head_pred_mean)
            head_pred_std=np.concatenate(head_pred_std)
            comp_values=np.concatenate(comp_values)

            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            error_diff = head_pred_mean - head_true
            pure_elements_dictionary = {'V': 23, 'Nb': 41, 'Ta': 73} #tmp for alloy datasets
            elements_list = ['V', 'Nb', 'Ta']

            for icomp in range(3):
                print(comp_values.shape)
                compvalue=comp_values[:,icomp].squeeze()
                ax = axs[irow, icomp]
                
                hist2d_norm = getcolordensity(compvalue, error_diff)
                sc=ax.scatter(compvalue, error_diff, s=12, c=hist2d_norm, vmin=0, vmax=1)
                ax.set_title(setname + "; " + varname, fontsize=24)
                if True: #irow==2:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    if irow==2:
                        fig.colorbar(sc, cax=cax, orientation='vertical')
                    else:
                        cax.set_axis_off()
                ax.set_box_aspect(1)
                ax.set_title(varname)
                if icomp==0:
                    ax.set_ylabel("Error=Pre-True")
                if irow==2:
                    ax.set_xlabel(f"Comp-{elements_list[icomp]}")
            
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.3)
    fig.savefig("./logs/" + log_name + f"/parity_plot_post_comp_{log_name}.png",dpi=500)
    fig.savefig("./logs/" + log_name + f"/parity_plot_post_comp_{log_name}.pdf")
    plt.close()
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
            comp_values = [None]*nprocs
            for iproc in range(nprocs):    
                file_name= saveresultsto +"head%d_proc%d.db"%(ihead, iproc)
                head_true[iproc], head_pred_mean[iproc], head_pred_std[iproc], comp_values[iproc] = get_ensemble_mean_std(file_name)
            head_true=np.concatenate(head_true)
            head_pred_mean=np.concatenate(head_pred_mean)
            head_pred_std=np.concatenate(head_pred_std)
            comp_values=np.concatenate(comp_values)

            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            error_diff = head_pred_mean - head_true
            pure_elements_dictionary = {'V': 23, 'Nb': 41, 'Ta': 73} #tmp for alloy datasets
            elements_list = ['V', 'Nb', 'Ta']
            ax = axs[icol]
            
            compvalue=np.sqrt(np.sum((comp_values-np.array([[0.5, 0.5, 0.5]]))**2, axis=1))
            hist2d_norm = getcolordensity(compvalue, error_diff)
            sc=ax.scatter(compvalue, error_diff, s=12, c=hist2d_norm, vmin=0, vmax=1)
            ax.set_title(setname + "; " + varname, fontsize=24)
            if True: #icol==2:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                if icol==2:
                    fig.colorbar(sc, cax=cax, orientation='vertical')
                else:
                    cax.set_axis_off()
            ax.set_box_aspect(1)
            ax.set_title(varname)
            if icol==0:
                ax.set_ylabel("Error=Pre-True")
            ax.set_xlabel(f"Comp-Dist")
        
    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.95, wspace=0.2, hspace=0.3)
    fig.savefig("./logs/" + log_name + f"/parity_plot_post_compdist_{log_name}.png",dpi=500)
    fig.savefig("./logs/" + log_name + f"/parity_plot_post_compdist_{log_name}.pdf")
    plt.close()
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
            comp_values = [None]*nprocs
            for iproc in range(nprocs):    
                file_name= saveresultsto +"head%d_proc%d.db"%(ihead, iproc)
                head_true[iproc], head_pred_mean[iproc], head_pred_std[iproc], comp_values[iproc] = get_ensemble_mean_std(file_name)
            head_true=np.concatenate(head_true)
            head_pred_mean=np.concatenate(head_pred_mean)
            head_pred_std=np.concatenate(head_pred_std)
            comp_values=np.concatenate(comp_values)

            if varname=="formation_enthalpy":
                print(f"T/V/T checking, {varname} outlier values:", head_true[np.where(head_true>=75)])
                print(f"T/V/T checking, {varname} outlier values with composition:", comp_values[np.where(head_true>=75)])
                head_pred_mean=head_pred_mean[np.where(head_true<75)]
                head_pred_std=head_pred_std[np.where(head_true<75)]
                comp_values=comp_values[np.where(head_true<75)]
                head_true=head_true[np.where(head_true<75)]
            elif varname=="rmsd":
                print(f"T/V/T checking, {varname} outlier values:", head_true[np.where(head_true<=1e-4)])
                print(f"T/V/T checking, {varname} outlier values with composition:", comp_values[np.where(head_true<=1e-4)])
                head_pred_mean=head_pred_mean[np.where(head_true>1e-4)]
                head_pred_std=head_pred_std[np.where(head_true>1e-4)]
                comp_values=comp_values[np.where(head_true>1e-4)]
                head_true=head_true[np.where(head_true>1e-4)]


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
    #fig.savefig("./logs/" + log_name + f"/parity_plot_post_errorbar_{log_name}.png",dpi=500)
    #fig.savefig("./logs/" + log_name + f"/parity_plot_post_errorbar_{log_name}.pdf")
    fig.savefig("./logs/" + log_name + f"/parity_plot_post_errorbar_{log_name}_.png",dpi=500)
    fig.savefig("./logs/" + log_name + f"/parity_plot_post_errorbar_{log_name}_.pdf")
    plt.close()
    ##################################################################################################################
    ##################################################################################################################
    linestyles=["-","--",":"]
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
            comp_values = [None]*nprocs
            for iproc in range(nprocs):    
                file_name= saveresultsto +"head%d_proc%d.db"%(ihead, iproc)
                head_true[iproc], head_pred_mean[iproc], head_pred_std[iproc], comp_values[iproc] = get_ensemble_mean_std(file_name)
            head_true=np.concatenate(head_true)
            head_pred_mean=np.concatenate(head_pred_mean)
            head_pred_std=np.concatenate(head_pred_std)
            comp_values=np.concatenate(comp_values)

            if varname=="formation_enthalpy":
                print(f"Uncertainty checking, {varname} outlier values:", head_true[np.where(head_true>=75)])
                print(f"Uncertainty checking, {varname} outlier values with composition:", comp_values[np.where(head_true>=75)])
                head_pred_mean=head_pred_mean[np.where(head_true<75)]
                head_pred_std=head_pred_std[np.where(head_true<75)]
                comp_values=comp_values[np.where(head_true<75)]
                head_true=head_true[np.where(head_true<75)]
            elif varname=="rmsd":
                print(f"Uncertainty checking, {varname} outlier values:", head_true[np.where(head_true<=1e-4)])
                print(f"Uncertainty checking, {varname} outlier values with composition:", comp_values[np.where(head_true<=1e-4)])
                head_pred_mean=head_pred_mean[np.where(head_true>1e-4)]
                head_pred_std=head_pred_std[np.where(head_true>1e-4)]
                comp_values=comp_values[np.where(head_true>1e-4)]
                head_true=head_true[np.where(head_true>1e-4)]

            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]
            
            hist1d, bin_edges = np.histogram(head_pred_std, bins=50)
            #_, bins = np.histogram(np.log10(head_pred_std), bins='auto')
            #hist1d, bin_edges = np.histogram(head_pred_std, bins=10**bins)

            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d/sum(hist1d), linestyles[icol], label=setname+" ("+str(sum(hist1d))+")", linewidth=2.0)
            np.savez("./logs/" + log_name +'/uncertainty.npz', uncer_bins=0.5 * (bin_edges[:-1] + bin_edges[1:]), count_ratio=hist1d/sum(hist1d))

        ax.set_title(varname, fontsize=24)
        ax.set_ylabel("Count Ratio", fontsize=28)
        ax.set_xlabel("Uncertainties")
         
    ax.legend()
    plt.subplots_adjust(left=0.25, bottom=0.15, right=0.98, top=0.925)#, wspace=0.2, hspace=0.3)
    #fig.savefig("./logs/" + log_name + f"/hist_uncertainty_{log_name}.png",dpi=500)
    #fig.savefig("./logs/" + log_name + f"/hist_uncertainty_{log_name}.pdf")
    fig.savefig("./logs/" + log_name + f"/hist_uncertainty_{log_name}_.png",dpi=500)
    fig.savefig("./logs/" + log_name + f"/hist_uncertainty_{log_name}_.pdf")
    plt.close()
    ##################################################################################################################
    ##################################################################################################################
    sys.exit(0)
