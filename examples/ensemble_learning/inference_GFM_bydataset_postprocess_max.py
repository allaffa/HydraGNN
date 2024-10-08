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
    parser.add_argument("--models_dir_folder", help="folder of trained models", type=str, default="examples/ensemble_learning/GFM_logs")
    parser.add_argument("--log", help="log name", default="GFM_EnsembleInference")
    parser.add_argument("--multi_model_list", help="multidataset list", default="ANI1x-v2,MPTrj-v2,OC2020-20M-v2,OC2022-v2,qm7x-v2")
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
        #print(head_pred_ens.size(), x_atomnums.size(), graph_batch.size())
        #print(x_atomnums[:50], graph_batch[:50])
        head_pred_mean = head_pred_ens.mean(axis=0)
        head_pred_std = head_pred_ens.std(axis=0)
        head_true = true_values.cpu().squeeze().numpy() 
        head_pred_ens = head_pred_ens.cpu().squeeze().numpy() 
        head_pred_mean = head_pred_mean.cpu().squeeze().numpy() 
        head_pred_std = head_pred_std.cpu().squeeze().numpy() 
        x_atomnums=x_atomnums.cpu().squeeze().numpy()
        graph_batch=graph_batch.cpu().squeeze().numpy()
        return head_true, head_pred_mean, head_pred_std, x_atomnums, graph_batch
    ##################################################################################################################
    nheads = len(config["NeuralNetwork"]["Variables_of_interest"]["output_names"])
    ##################################################################################################################
    for icol, setname in enumerate(["train", "val", "test"]):
        for idataset, dataset in enumerate(modellist):
            saveresultsto=f"./logs/{log_name}/{dataset}_{setname}_"
            for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
            config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
            config["NeuralNetwork"]["Variables_of_interest"]["type"],
            config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
            )):
                file_name= saveresultsto +"head%d_atomnum_batch_0.db"%ihead
                head_true, head_pred_ens, head_pred_uncertainty, x_atomnums, graph_batch = get_ensemble_mean_std(file_name)
                ifeat = var_config["output_index"][ihead]
                outtype = var_config["type"][ihead]
                varname = var_config["output_names"][ihead]
                print("==========================================================")
                print(setname, " head %d"%ihead, varname)
                #shape of head true
                print("shape of head true",head_true.shape)
                #shape of head ensemble predict
                print("shape of head ensemble predict",head_pred_ens.shape)
                #shape of head uncertainty
                print("shape of head uncertainty", head_pred_uncertainty.shape)
                #shape of graph atomic numbers
                print("shape of graph atomic numbers",x_atomnums.shape)
                #shape of graph batch
                print("shape of graph batch",graph_batch.shape)
                print("==========================================================")
            
    
   
