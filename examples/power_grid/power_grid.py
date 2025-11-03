import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import random

import pandas as pd

import numpy as np

from scipy.sparse import csc_matrix

import torch
import torch.distributed as dist
from torch_geometric.data import Data

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.input_config_parsing.config_utils import get_log_name_config
from hydragnn.utils.model import print_model

from hydragnn.utils.distributed import nsplit

from hydragnn.preprocess.load_data import split_dataset

import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.preprocess.graph_samples_checks_and_updates import RadiusGraph

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from hydragnn.utils.print.print_utils import log

from hydragnn.utils.distributed import nsplit

# FIX random seed
random_state = 0
torch.manual_seed(random_state)


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def rescale(value, old_min=0.9, old_max=1.1, new_min=0.0, new_max=1.0):
    return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


"""
data.x[:,0] contains P
data.x[:,1] contains Q
data.x[:,2] contains Vmag (magnitude of voltage)
data.x[:,3] contains theta (phase angle of voltage measured in radiants)
data.x[:,4] is binary columns. 1 means that P is known, 0 means that P is not known
data.x[:,5] is binary columns. 1 means that Q is known, 0 means that Q is not known
data.x[:,6] is binary columns. 1 means that V is known, 0 means that V is not known
data.x[:,7] is binary columns. 1 means that theta is known, 0 means that theta is not known

data.y[:,0] contains values of Vmag (magnitude of voltage)
data.y[:,1] contains values of theta (phase angle of voltage measured in radiants)
"""


class PowerGridDataset(AbstractBaseDataset):
    def __init__(self, dirpath, var_config, dist=False):
        super().__init__()

        self.var_config = var_config
        self.data_path = os.path.join(dirpath, "output_files")

        self.radius_graph = RadiusGraph(5.0, loop=False, max_num_neighbors=50)

        """
        Input features: 
        For each PQ node: we know Pin and Qin. 
        For each PV node : we know P and Vmag. 
        for Slack bus: we know Vmag and Vangle. 
        """

        # TO DO: assume you know Pin and Qin at generation and load buses, goal: predict Vmag and Vangle at all buses

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        directories_list = [
            d
            for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ]

        for directory in directories_list:

            csv_files_list = [
                file
                for file in os.listdir(os.path.join(self.data_path, directory))
                if file.endswith(".csv")
            ]

            local_csv_files_list = list(nsplit(csv_files_list, self.world_size))[
                self.rank
            ]
            log("local files list", len(local_csv_files_list))

            for csv_file in local_csv_files_list:

                # Extract the string "Instance"
                base_name = os.path.splitext(csv_file)[0]  # Remove the file extension

                # per unit scaling factor
                # Specify the file name
                per_unit_scale_factor_name = os.path.join(
                    self.data_path,
                    directory,
                    base_name + "_per_unit_scaling_factor.txt",
                )

                # Open the file in read mode and read the float
                with open(per_unit_scale_factor_name, "r") as file:
                    content = (
                        file.read().strip()
                    )  # Read the content and remove any extra whitespace or newlines
                    S_base = float(content)  # Convert the string to a float

                # Load the saved .npz file to extract information about the binary adjacency matrix
                loaded_data = np.load(
                    os.path.join(
                        self.data_path,
                        directory,
                        base_name + "_adjacency_binary_matrix.npz",
                    )
                )
                rows = loaded_data["rows"]
                cols = loaded_data["cols"]

                # Create edge_index
                edge_index = torch.tensor([rows, cols], dtype=torch.long)

                # Load the saved .npz file to extract information about the conductance matrix
                conductance_loaded_data = np.load(
                    os.path.join(
                        self.data_path, directory, base_name + "_conductance_matrix.npz"
                    )
                )

                # Reconstruct the csc_matrix for resistance
                G_loaded = csc_matrix(
                    (
                        conductance_loaded_data["data"],
                        conductance_loaded_data["indices"],
                        conductance_loaded_data["indptr"],
                    ),
                    shape=conductance_loaded_data["shape"],
                )

                # Create resistance as edge attribute
                conductance_attribute = torch.tensor(
                    G_loaded[rows, cols], dtype=torch.float32
                )

                # Load the saved .npz file to extract information about the conductance matrix
                susceptance_loaded_data = np.load(
                    os.path.join(
                        self.data_path, directory, base_name + "_susceptance_matrix.npz"
                    )
                )

                # Reconstruct the csc_matrix for resistance
                B_loaded = csc_matrix(
                    (
                        susceptance_loaded_data["data"],
                        susceptance_loaded_data["indices"],
                        susceptance_loaded_data["indptr"],
                    ),
                    shape=susceptance_loaded_data["shape"],
                )

                # Create resistance as edge attribute
                susceptance_attribute = torch.tensor(
                    B_loaded[rows, cols], dtype=torch.float32
                )

                # Load the saved .npz file to extract information about the resistance matrix
                resistance_loaded_data = np.load(
                    os.path.join(
                        self.data_path, directory, base_name + "_resistance_matrix.npz"
                    )
                )

                # Reconstruct the csc_matrix for resistance
                R_loaded = csc_matrix(
                    (
                        resistance_loaded_data["data"],
                        resistance_loaded_data["indices"],
                        resistance_loaded_data["indptr"],
                    ),
                    shape=resistance_loaded_data["shape"],
                )

                # Create resistance as edge attribute
                resistance_attribute = torch.tensor(
                    R_loaded[rows, cols], dtype=torch.float32
                )

                # Load the saved .npz file to extract information about the reactance matrix
                reactance_loaded_data = np.load(
                    os.path.join(
                        self.data_path, directory, base_name + "_reactance_matrix.npz"
                    )
                )

                # Reconstruct the csc_matrix for reactance
                X_loaded = csc_matrix(
                    (
                        reactance_loaded_data["data"],
                        reactance_loaded_data["indices"],
                        reactance_loaded_data["indptr"],
                    ),
                    shape=reactance_loaded_data["shape"],
                )

                # Create resistance as edge attribute
                reactance_attribute = torch.tensor(
                    X_loaded[rows, cols], dtype=torch.float32
                )

                # Check that binary adjacency matrix, resistance matrix, and reactance matrix have the same shape
                assert edge_index.shape[1] == conductance_attribute.shape[1]
                assert edge_index.shape[1] == susceptance_attribute.shape[1]
                assert edge_index.shape[1] == resistance_attribute.shape[1]
                assert edge_index.shape[1] == reactance_attribute.shape[1]

                edge_attr = torch.cat(
                    [
                        conductance_attribute,
                        susceptance_attribute,
                        resistance_attribute,
                        reactance_attribute,
                    ],
                    dim=0,
                ).t()

                # Load the CSV file into a DataFrame
                data_csv = pd.read_csv(
                    os.path.join(self.data_path, directory, csv_file)
                )
                node_features = torch.cat(
                    [
                        torch.tensor(data_csv["Pin"]).unsqueeze(1),
                        torch.tensor(data_csv["Qin"]).unsqueeze(1),
                        torch.tensor(data_csv["Vmag"]).unsqueeze(1),
                        torch.tensor(data_csv["Vang"]).unsqueeze(1),
                    ],
                    dim=1,
                )

                S_base_tensor = torch.ones(node_features.shape[0], 1, dtype=torch.float32) * S_base 

                # Apply rescale to Vmag (index 2)
                # node_features[:, 2] = rescale(node_features[:, 2], old_min=0.95, old_max=1.1)

                # Initialize an empty mask with the same shape as node_features
                node_mask = torch.zeros_like(node_features, dtype=torch.int)

                # Assign values to the mask based on 'Bus Type'
                for idx, bus_type in enumerate(data_csv["Bus Type"]):
                    if bus_type == "Slack":
                        node_mask[idx] = torch.tensor([0, 0, 1, 1], dtype=torch.int)
                    elif bus_type == "PQ":
                        node_mask[idx] = torch.tensor([1, 1, 0, 0], dtype=torch.int)
                    elif bus_type == "PV":
                        node_mask[idx] = torch.tensor([1, 0, 1, 0], dtype=torch.int)

                # De-activate input features of the nodes based on the partial information available in real-case scenarios
                node_features = node_features * node_mask

                # Provide information about what has been masked as additional inoptu feature
                node_features = torch.cat([node_features, node_mask], dim=1)

                # we need to concatenate the voltage feaurtes one more time because the HydraGNN code will extract them and put them in data.y
                node_features = torch.cat(
                    [
                        node_features,
                        torch.tensor(data_csv.Vmag).unsqueeze(1),
                        torch.tensor(data_csv.Vang).unsqueeze(1),
                    ],
                    dim=1,
                ).to(dtype=torch.float32)

                # Add mask as additional input
                # data_sample = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, grid_system=base_name)
                data_sample = Data(
                    per_unit_scaling_factor=S_base_tensor,
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    grid_system=base_name,
                    true_P=torch.tensor(data_csv["Pin"]).unsqueeze(1),
                    true_Q=torch.tensor(data_csv["Qin"]).unsqueeze(1),
                )

                self.dataset.append(data_sample)

        random.shuffle(self.dataset)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="power_grid.json"
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument("--modelname", help="model name")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="adios")
    args = parser.parse_args()

    graph_feature_names = [""]
    graph_feature_dims = []
    node_feature_names = [
        "Pin",
        "Qin",
        "Vmag",
        "Vang",
        "Mask1",
        "Mask2",
        "Mask3",
        "Mask4",
        "Vmag_output",
        "Vang_output",
    ]
    node_feature_dims = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "PowerGrid" if args.log is None else args.log
    hydragnn.utils.print.print_utils.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "PowerGrid" if args.modelname is None else args.modelname
    if args.preonly:
        ## local data
        total = PowerGridDataset(
            os.path.join(datadir),
            var_config,
            dist=True,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        setnames = ["trainset", "valset", "testset"]

        ## adios
        if args.format == "adios":
            fname = os.path.join(
                os.path.dirname(__file__), "./dataset/%s.bp" % modelname
            )
            adwriter = AdiosWriter(fname, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            # adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
            # adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
            adwriter.add_global("pna_deg", deg)
            adwriter.save()

        ## pickle
        elif args.format == "pickle":
            basedir = os.path.join(
                os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
            )
            attrs = dict()
            attrs["pna_deg"] = deg
            SimplePickleWriter(
                trainset,
                basedir,
                "trainset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                valset,
                basedir,
                "valset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
            SimplePickleWriter(
                testset,
                basedir,
                "testset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
        sys.exit(0)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    if args.format == "adios":
        info("Adios load")
        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
        # minmax_node_feature = trainset.minmax_node_feature
        # minmax_graph_feature = trainset.minmax_graph_feature
        pna_deg = trainset.pna_deg
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            # trainset.minmax_node_feature = minmax_node_feature
            # trainset.minmax_graph_feature = minmax_graph_feature
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    timer.stop()

    # Enable power grid PINN model wrapper
    config["NeuralNetwork"]["Architecture"]["enable_power_grid_pinn"] = True
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    # Print details of neural network architecture
    print_model(model)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=False,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)
