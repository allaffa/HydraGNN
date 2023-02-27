import os, json
import matplotlib.pyplot as plt
import random
import pickle, csv
import pandas as pd

import logging
import sys
from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
import argparse
import time

import torch_geometric

import hydragnn
from hydragnn.utils.distributed import setup_ddp
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.model import print_model, load_existing_model
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_smilestr,
)
from hydragnn.preprocess.utils import get_radius_graph

from hydragnn.models.create import create_model_config
from hydragnn.utils.config_utils import (
    update_config,
    get_log_name_config,
)

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist

import warnings

warnings.filterwarnings("error")

import time

plt.rcParams.update({"font.size": 16})
#########################################################

CaseDir = os.path.join(
    os.path.dirname(__file__),
    "logs",
)

DatasetDir = os.path.join(
    os.path.dirname(__file__),
    "dataset",
)

maximum_input = torch.load('maximum_input.pt')

######################################################################
for irun in range(1, 2):

    number_data_samples = 1000
    start_time = 0.0
    finish_time = 0.0

    for icase in range(1):
        config_file = CaseDir + "/concrete_shielding_fullx/" + "config"
        with open(config_file + ".json", "r") as f:
            config = json.load(f)

        os.environ["SERIALIZED_DATA_PATH"] = CaseDir

        world_size, world_rank = setup_ddp()

        pickle_path = DatasetDir + "/pickle/"

        normalized_trainset = SimplePickleDataset(pickle_path, "concrete_shielding", "trainset")
        normalized_valset = SimplePickleDataset(pickle_path, "concrete_shielding", "valset")
        normalized_testset = SimplePickleDataset(pickle_path, "concrete_shielding", "testset")

        (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
            normalized_trainset, normalized_valset, normalized_testset,
            config["NeuralNetwork"]["Training"]["batch_size"]
        )

        config = update_config(config, train_loader, val_loader, test_loader)

        model = create_model_config(
            config=config["NeuralNetwork"],
            verbosity=config["Verbosity"]["level"],
        )

        model = torch.nn.parallel.DistributedDataParallel(
            model
        )

        model_name = "concrete_shielding_fullx"

        load_existing_model(model, model_name, path="./logs/")
        list_indices_large_mismatch_expansion = []
        time_step_index_large_mismatch_expansion = []
        list_indices_large_mismatch_damage = []
        time_step_index_large_mismatch_damage = []

        for test_data in test_loader.dataset:
            test_data.x = torch.matmul(test_data.x, torch.diag(1. / maximum_input))
            prediction = model(test_data)
            difference_average_linear_expansion = test_data.y[0:1681] - prediction[0]
            difference_average_damage = test_data.y[1681:] - prediction[1]

            if max(abs(difference_average_linear_expansion)) > 0.3:
                indices_expansion = torch.Tensor([item for item in range(0, 1681)]).unsqueeze(1)
                new_list = list(indices_expansion[abs(difference_average_linear_expansion) > 0.3])
                list_indices_large_mismatch_expansion.extend(new_list)
                time_step_index_large_mismatch_expansion.extend([test_data.time_step_index]*len(new_list))

            if max(abs(difference_average_damage)) > 0.3:
                indices_damage = torch.Tensor([item for item in range(0,1681)]).unsqueeze(1)
                new_list = list(indices_damage[abs(difference_average_damage) > 0.3])
                list_indices_large_mismatch_damage.extend(new_list)
                time_step_index_large_mismatch_damage.extend([test_data.time_step_index] * len(new_list))

        assert len(list_indices_large_mismatch_expansion) == len(time_step_index_large_mismatch_expansion)
        assert len(list_indices_large_mismatch_damage) == len(time_step_index_large_mismatch_damage)

        # create histogram from list of data
        plt.figure()
        plt.hist(list_indices_large_mismatch_expansion, bins=1681)
        plt.savefig('expansion.png')
        plt.close()

        # create histogram from list of data
        plt.figure()
        plt.hist(list_indices_large_mismatch_damage, bins=1681)
        plt.savefig('damage.png')
        plt.close()

        with open("expansion.txt", "w") as expansion_file:
            expansion_file.write("Node index \t Time \n")
            for index in range(0, len(list_indices_large_mismatch_expansion)):
                expansion_file.write( str(list_indices_large_mismatch_expansion[index])+"\t"+str(time_step_index_large_mismatch_expansion[index])+"\n" )
            expansion_file.close()

        with open("damage.txt", "w") as damage_file:
            damage_file.write("Node index \t Time \n")
            for index in range(0, len(list_indices_large_mismatch_damage)):
                damage_file.write( str(list_indices_large_mismatch_damage[index])+"\t"+str(time_step_index_large_mismatch_damage[index])+"\n" )
            damage_file.close()




