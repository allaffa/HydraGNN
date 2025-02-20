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
import ternary
import matplotlib.cm as cm
from matplotlib.colors import Normalize
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
    def plot_ternary_field(ax, points, values, labels, varname, error_mae):
        """
        Plot the ternary diagram field using the maximum values from the compositions.
        Handles cases with fewer than 3 atom types.
        """
        # Initialize the ternary plot
        scale = 1.0
        _, tax = ternary.figure(ax=ax, scale=scale)
        tax.boundary(linewidth=2.0)

        # Remove the external box by disabling axes spines
        tax.clear_matplotlib_ticks()
        for spine in tax.get_axes().spines.values():
            spine.set_visible(False)

        # Customize gridlines
        tax.gridlines(multiple=0.1, color="gray", linewidth=0.5)

        # Set tick formats for each axis to display one decimal place
        tick_formats = {
            'b': "%.1f",  # Bottom axis
            'l': "%.1f",  # Left axis
            'r': "%.1f"  # Right axis
        }
        tax.ticks(axis='lbr', multiple=0.1, linewidth=1, offset=0.02, tick_formats=tick_formats, fontsize=12)

        # Set the title
        # tax.set_title("Maximum MAE", fontsize=15)
        vmin = min(values) #0.0
        vmax = max(values)

        cmap = cm.get_cmap("viridis")  # Select colormap

        # Plot each composition with color intensity based on the value
        unique_comp={}
        for point, value in zip(points, values):
            key = "-".join([str(item) for item in list(point)]) 
            if key not in unique_comp:
                unique_comp[key]=[]
            unique_comp[key].append(value)
        points_unq=[]
        vmean_unq=[]
        vstd_unq=[]
        for key in unique_comp:
            points_unq.append([float(item) for item in key.split("-")])
            vmean_unq.append(sum(unique_comp[key])/len(unique_comp[key]))
        print(points_unq, vmean_unq)
        tax.scatter(points_unq, c=vmean_unq, marker='o', vmin=vmin, vmax=vmax, cmap=cmap)
        tax.get_axes().set_aspect('equal', adjustable='box')

        # Add color bar to the right
        # Create a normalization object with the desired range
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Create a color bar with the specified normalization
        cbar=plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=tax.get_axes(),
            orientation="vertical",
            shrink=0.5, pad=0.04#pad=0.1
        )
        cbar.ax.tick_params(labelsize=14)
        # Set corner labels
        tax.left_axis_label(labels[0], fontsize=18, offset=0.15)   # Component A
        tax.right_axis_label(labels[1], fontsize=18, offset=0.15)  # Component B
        tax.bottom_axis_label(labels[2], fontsize=18, offset=0.15) # Component C
        # Force redraw of labels to ensure they appear
        tax._redraw_labels()
        
        #tax.set_title(varname+"; MAE={:.2e}".format(error_mae), fontsize=16) 

        return ax

        # Save the plot to a .png file
        #plt.savefig(output_file, dpi=300)

    ##################################################################################################################
    for icol, setname in enumerate(["test"]):
        saveresultsto=f"./logs/{log_name}/{setname}_"

        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            assert ihead==0 and icol==0
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
                print(f"checking {varname} outlier values:", head_true[np.where(head_true>=75)])
                print(f"checking {varname} outlier values with composition:", comp_values[np.where(head_true>=75)])
                head_pred_mean=head_pred_mean[np.where(head_true<75)]
                head_pred_std=head_pred_std[np.where(head_true<75)]
                comp_values=comp_values[np.where(head_true<75)]
                head_true=head_true[np.where(head_true<75)]
            elif varname=="rmsd":
                print(np.where(head_true<=1e-4))
                print(f"checking {varname} outlier values:", head_true[np.where(head_true<=1e-4)])
                print(f"checking {varname} outlier values with composition:", comp_values[np.where(head_true<=1e-4)])
                head_pred_mean=head_pred_mean[np.where(head_true>1e-4)]
                head_pred_std=head_pred_std[np.where(head_true>1e-4)]
                comp_values=comp_values[np.where(head_true>1e-4)]
                head_true=head_true[np.where(head_true>1e-4)]

            #error_diff = head_pred_mean - head_true
            error_diff = np.abs(head_pred_mean - head_true)
            error_mae = np.mean(np.abs(head_pred_mean - head_true))

            pure_elements_dictionary = {'V': 23, 'Nb': 41, 'Ta': 73} #tmp for alloy datasets
            elements_list = ['V', 'Nb', 'Ta']
            #points = []
            #for isamp in range(len(comp_values)):
            #    points.append(comp_values[isamp,:])
            fig, axs = plt.subplots(1,1, figsize=(5, 5))
            ax=plot_ternary_field(axs, comp_values, error_diff, elements_list, varname, error_mae)
            """
            for icomp in range(3):
                print(comp_values.shape)
                compvalue=comp_values[:,icomp].squeeze()
                ax=axs[icomp]
                hist2d_norm = getcolordensity( compvalue, error_diff,)

                sc=ax.scatter(compvalue, error_diff, s=12, c=hist2d_norm, vmin=0, vmax=1)
                if True: #icol==2:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cax.set_axis_off()
                ax.set_box_aspect(1)
                ax.set_title(varname)
                if icomp==0:
                    ax.set_ylabel("Error=Pre-True")
                ax.set_xlabel(f"Comp-{elements_list[icomp]}")
            """
                    
    ##################################################################################################################
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95)#, wspace=0.15, hspace=0.3)
    fig.savefig("./logs/" + log_name + f"/crosscheck_post_ternaryplot_{log_name}.png",dpi=500)
    fig.savefig("./logs/" + log_name + f"/crosscheck_post_ternaryplot_{log_name}.pdf")
    plt.close()
    sys.exit(0)
