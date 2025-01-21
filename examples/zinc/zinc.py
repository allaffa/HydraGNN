import sys
import logging
import os, json

from mpi4py import MPI

import argparse

import torch
import torch_geometric
from torch_geometric.datasets import ZINC
import torch_geometric.transforms as T
from torch_geometric.transforms import AddLaplacianEigenvectorPE

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
import hydragnn.utils.profiling_and_tracing.tracer as tr

# Set this path for output.
try:
    os.environ["SERIALIZED_DATA_PATH"]
except:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

from hydragnn.utils.print.print_utils import log

from hydragnn.utils.distributed import nsplit

# FIX random seed
random_state = 0
torch.manual_seed(random_state)


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zinc.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

log_name = "zinc_test"
# Enable print to log file.
hydragnn.utils.print.print_utils.setup_log(log_name)

# Use built-in torch_geometric dataset.
# NOTE: data is moved to the device in the pre-transform.
# NOTE: transforms/filters will NOT be re-run unless the zinc/processed/ directory is removed.
lapPE = AddLaplacianEigenvectorPE(
    k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
    attr_name="pe",
    is_undirected=True,
)


def zinc_pre_transform(data):
    data.x = data.x.float().view(-1, 1)
    data.edge_attr = data.edge_attr.float().view(-1, 1)
    data = lapPE(data)
    # gps requires relative edge features, introduced rel_lapPe as edge encodings
    source_pe = data.pe[data.edge_index[0]]
    target_pe = data.pe[data.edge_index[1]]
    data.rel_pe = torch.abs(source_pe - target_pe)  # Compute feature-wise difference
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument("--inputfile", help="input file", type=str, default="zinc.json")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--modelname", help="model name")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )

    parser.set_defaults(format="pickle")
    args = parser.parse_args()
    args.parameters = vars(args)

    graph_feature_names = ["free_energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number"]
    node_feature_dims = [1]

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

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "ZINC" if args.log is None else args.log
    hydragnn.utils.print.print_utils.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "ZINC" if args.modelname is None else args.modelname

    if args.preonly:

        trainset = ZINC(
            root="dataset/zinc",
            subset=False,
            split="train",
            pre_transform=zinc_pre_transform,  # TODO:change subset=True before merge
        )
        valset = ZINC(
            root="dataset/zinc",
            subset=False,
            split="val",
            pre_transform=zinc_pre_transform,  # TODO:change subset=True before merge
        )
        testset = ZINC(
            root="dataset/zinc",
            subset=False,
            split="test",
            pre_transform=zinc_pre_transform,  # TODO:change subset=True before merge
        )

        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        # Create global list of indices for data samples
        train_indices = list(range(0, len(trainset)))
        val_indices = list(range(0, len(valset)))
        test_indices = list(range(0, len(testset)))

        # Partition index of data samples across distributed processes
        local_train_indices = list(nsplit(train_indices, comm_size))[rank]
        local_val_indices = list(nsplit(val_indices, comm_size))[rank]
        local_test_indices = list(nsplit(test_indices, comm_size))[rank]

        # Convert datasets into lists of torch_geometric.data.Data objects
        train_list = [trainset[index] for index in local_train_indices]
        val_list = [valset[index] for index in local_val_indices]
        test_list = [testset[index] for index in local_test_indices]

        deg = gather_deg(train_list)
        config["pna_deg"] = deg

        ## pickle
        if args.format == "pickle":
            basedir = os.path.join(
                os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
            )
            attrs = dict()
            attrs["pna_deg"] = deg
            SimplePickleWriter(
                train_list,
                basedir,
                "trainset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                val_list,
                basedir,
                "valset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
            SimplePickleWriter(
                test_list,
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

    if args.format == "pickle":
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
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

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

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.model.model.load_existing_model_config(
        model=model, config=config["NeuralNetwork"]["Training"]
    )

    # Run training with the given model and zinc dataset.
    writer = hydragnn.utils.model.model.get_summary_writer(log_name)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

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
