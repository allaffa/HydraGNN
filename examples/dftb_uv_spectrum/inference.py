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
import mpi4py
mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False    # do not finalize MPI automatically
from mpi4py import MPI
MPI.Init()


import argparse
import os, json
import matplotlib.pyplot as plt
import pickle

import numpy as np
from tqdm import tqdm

from hydragnn.models.create import create_model_config
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.utils.distributed import setup_ddp, get_device
from hydragnn.utils.model import load_existing_model

from hydragnn.utils.pickledataset import SimplePickleDataset

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch


parser = argparse.ArgumentParser()
parser.add_argument(
    "--w_nm",
    default = 10.0,
    type = float,
    help="wavelength of the dataset to load",
)
args = parser.parse_args()

#-------------------------------------------------
# Plot Settings
#-------------------------------------------------
plt.style.use('bmh')
plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams["font.size"] = 45
# # plt.rcParams["font.weight"] = 'bold'
plt.rcParams["xtick.color"] = 'black'
plt.rcParams["ytick.color"] = 'black'
plt.rcParams["axes.edgecolor"] = 'black'
plt.rcParams["axes.linewidth"] = 1
figure_dpi = 300  # DPI of the picture
plt_y_lim = 0.45
# x-axis
lower = 0.0
upper = 750.0
length = 37500
plt_x_lim = 37500
tick_count = 11
x_ticks = list(np.linspace(0,length,tick_count))
x_labels = [int(val) for val in np.linspace(lower,upper,tick_count)]

#########################################################


modelname = "dftb_smooth_uv_spectrum"

CaseDir = os.path.join(
    os.path.dirname(__file__),
    "logs",
)

DatasetDir = os.path.join(
    os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
)



######################################################################
for irun in range(1, 2):

    number_data_samples = 1000
    start_time = 0.0
    finish_time = 0.0

    os.environ["SERIALIZED_DATA_PATH"] = CaseDir

    world_size, world_rank = setup_ddp()

    pickle_path = DatasetDir + ""

    log_name = './logs/dftb_smooth_uv_spectrum'
    config_file = f'{log_name}/config'
    with open(config_file + ".json", "r") as f:
        config = json.load(f)

    ##set directory to load processed pickle files, train/validate/test
    testset = SimplePickleDataset(pickle_path, "testset", var_config=config["NeuralNetwork"]["Variables_of_interest"])

    model = create_model_config(
        config=config["NeuralNetwork"],
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model
    )

    load_existing_model(model, modelname, path="./logs/")

    num_test_samples = len(testset)
    mol_ID = [item.ID for item in testset]

    FRQ=1
    error_mae, error_mse = [], []
    for sample_id, test_data in enumerate(tqdm(testset)):
        prediction = model(test_data.to(get_device()))[0].squeeze().detach().cpu().numpy()
        prediction = np.expand_dims(prediction, axis=1)
        true = test_data.y.detach().to('cpu').numpy()

        error_mae += [np.sum(np.abs(true - prediction))]
        error_mse += [np.sqrt(np.sum(true - prediction)**2/true.shape[0])]

    #-------------------------------------------------
    # MSE
    #-------------------------------------------------
    mean = np.mean(error_mse)
    std = np.std(error_mse)
    print(f'SE range: ({min(error_mse):.4f},{max(error_mse):.4f})')
    print(f'SE mean,std: ({mean:.3f},{std:.3f})')

    model_name = config["NeuralNetwork"]["Architecture"]["model_type"]
    fig, ax = plt.subplots(1, 1)
    ax.hist(error_mse, bins=20, edgecolor='black', linewidth=3)
    plt.xlim(0,6.0)
    plt.xticks([1.0,3.0,5.0])
    plt.yticks([1000,2000,3000,4000])
    plt.grid(True)
    plt.xlabel('Squared Error', color='black')
    plt.ylabel('# Molecules', color='black')
    plt.legend()
    plt.title(f'{model_name} $\mu$:{mean:.3f}, $\sigma$:{std:.3f}')
    plt.draw()
    plt.tight_layout()
    plt.savefig(f"logs/{model_name}_{args.w_nm}nm_mse_bin.png")
    plt.close(fig)

    mse_min_idx = np.argmin(error_mse)
    mse_max_idx = np.argmax(error_mse)
    mse_med_idx =  np.abs(error_mse - np.median(error_mse)).argmin()
    indices = [mse_min_idx, mse_med_idx, mse_max_idx]
    data = [testset[index] for index in indices]
    idx_strings = ['minimum', 'median', 'maximum']

    for (idx, idx_str, test_data) in zip(indices[:10], idx_strings[:10], data[:10]):
        print(idx_str, error_mae[idx])
        prediction = model(test_data.to(get_device()))[0].squeeze().detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1)
        ax.plot(bins[::FRQ], test_data.y.detach().to('cpu')[::FRQ], color='black', label="TD-DFTB", linewidth=5, dashes=[1,0])
        ax.plot(bins[::FRQ], prediction[::FRQ], label=f"{model_name}", color='red', linewidth=5, dashes=[4,2])
        plt.title("molecule ID: "+mol_ID[idx])
        plt.legend(fancybox=True,handlelength=1,shadow=True,loc='upper right',bbox_to_anchor=(1.025,1.0),ncol=1)
        plt.draw()
        plt.tight_layout()
        plt.xlim([0,plt_x_lim])
        plt.xticks(x_ticks, x_labels)
        # plt.ylim([-0.2, max(true_values_AE)+0.2])
        plt.savefig(f"logs/{model_name}_{args.w_nm}nm_mse_{idx_str}.png")
        plt.close(fig)

    #-------------------------------------------------
    # MAE
    #-------------------------------------------------
    mean = np.mean(error_mae)
    std = np.std(error_mae)
    print(f'AE range: ({min(error_mae),max(error_mae)})')
    print(f'AE mean,std: ({mean:.3f},{std:.3f})')
    fig, ax = plt.subplots(1, 1)
    ax.hist(error_mae, bins=14)
    plt.legend()
    plt.title(f'{model_name} $\mu$:{mean:.3f}, $\sigma$:{std:.3f}')
    plt.draw()
    plt.tight_layout()
    plt.savefig(f"logs/{model_name}_{args.w_nm}nm_mae_bin.png")
    plt.close(fig)

    mae_min_idx = np.argmin(error_mae)
    mae_max_idx = np.argmax(error_mae)
    mae_med_idx =  np.abs(error_mae - np.median(error_mae)).argmin()
    indices = [mae_min_idx, mae_med_idx, mae_max_idx, 6]
    data = [testset[index] for index in indices]
    idx_strings = ['minimum', 'median', 'maximum', 'fixed']

    for (idx, idx_str, test_data) in zip(indices, idx_strings, data):
        print(idx_str, error_mae[idx])
        prediction = model(test_data.to(get_device()))[0].squeeze().detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1)
        ax.plot(bins[::FRQ], test_data.y.detach().to('cpu')[::FRQ], color='black', label="TD-DFTB", linewidth=5, dashes=[1,0])
        ax.plot(bins[::FRQ], prediction[::FRQ], label=f"{model_name}", color='red', linewidth=5, dashes=[4,2])
        plt.title("molecule ID: "+mol_ID[idx])
        plt.legend(fancybox=True,handlelength=1,shadow=True,loc='upper right',bbox_to_anchor=(1.025,1.0),ncol=1)
        plt.draw()
        plt.tight_layout()
        plt.xlim([0,plt_x_lim])
        plt.xticks(x_ticks, x_labels)
        # plt.ylim([-0.2, max(true_values_AE)+0.2])
        plt.savefig(f"logs/{model_name}_{args.w_nm}nm_mae_{idx_str}.png")
        plt.close(fig)

MPI.Finalize()
