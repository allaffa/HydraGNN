import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import torch
import numpy as np
from torch_geometric.datasets import ZINC
import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.print.print_utils import log

try:
    from hydragnn.utils.adiosdataset import AdiosDataset
except ImportError:
    pass

from scipy.interpolate import BSpline, make_interp_spline
import adios2 as ad2

## FIMME
torch.backends.cudnn.enabled = False


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def zinc_pre_transform(data, transform):
    data.x = data.x.float().view(-1, 1)
    data.edge_attr = data.edge_attr.float().view(-1, 1)
    data = transform(data)
    # gps requires relative edge features, introduced rel_lapPe as edge encodings
    source_pe = data.pe[data.edge_index[0]]
    target_pe = data.pe[data.edge_index[1]]
    data.rel_pe = torch.abs(source_pe - target_pe)  # Compute feature-wise difference
    return data


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--inputfile", help="input file", type=str, default="zinc.json")
    parser.add_argument("--mpnn_type", help="mpnn_type", default="PNA")
    parser.add_argument("--hidden_dim", type=int, help="hidden_dim", default=64)
    parser.add_argument(
        "--num_conv_layers", type=int, help="num_conv_layers", default=2
    )
    parser.add_argument("--num_headlayers", type=int, help="num_headlayers", default=2)
    parser.add_argument("--dim_headlayers", type=int, help="dim_headlayers", default=10)
    parser.add_argument("--global_attn_heads", help="global_attn_heads", default=None)
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name", default="zinc_hpo_trials")
    parser.add_argument("--num_epoch", type=int, help="num_epoch", default=None)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument(
        "--multi_model_list", help="multidataset list", default="OC2020"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="set num samples per process for weak-scaling test",
        default=None,
    )
    parser.add_argument(
        "--compute_grad_energy",
        action="store_true",
        help="use automatic differentiation to compute gradiens of energy",
        default=False,
    )

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
    group.add_argument(
        "--multi",
        help="Multi dataset",
        action="store_const",
        dest="format",
        const="multi",
    )
    parser.set_defaults(format="adios")
    args = parser.parse_args()
    args.parameters = vars(args)

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates", "forces"]
    node_feature_dims = [1, 3, 3]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
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

    if args.parameters["global_attn_heads"] is not None:
        config["NeuralNetwork"]["Architecture"]["global_attn_heads"] = args.parameters[
            "global_attn_heads"
        ]
        global_attn_heads = args.parameters["global_attn_heads"]
        hidden_dim = global_attn_heads * args.parameters["hidden_dim"]
    else:
        hidden_dim = args.parameters["hidden_dim"]

    # Update the config dictionary with the suggested hyperparameters
    config["NeuralNetwork"]["Architecture"]["mpnn_type"] = args.parameters["mpnn_type"]
    config["NeuralNetwork"]["Architecture"]["hidden_dim"] = hidden_dim
    config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = args.parameters[
        "num_conv_layers"
    ]

    dim_headlayers = [
        trial.parameters["dim_headlayers"]
        for i in range(trial.parameters["num_headlayers"])
    ]

    for head_type in trial_config["NeuralNetwork"]["Architecture"]["output_heads"]:
        trial_config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
            "num_headlayers"
        ] = trial.parameters["num_headlayers"]
        trial_config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
            "dim_headlayers"
        ] = dim_headlayers

    config["NeuralNetwork"]["Architecture"]["equivariance"] = False

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch

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

    log_name = "zinc_deephyper_trials" if args.log is None else args.log
    hydragnn.utils.print.print_utils.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zinc.json")
    with open(filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]

    # LPE
    transform = AddLaplacianEigenvectorPE(
        k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
        attr_name="pe",
        is_undirected=True,
    )

    # Use built-in torch_geometric datasets.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the zinc/processed/ directory is removed.
    trainset = ZINC(
        root="dataset/zinc",
        subset=False,
        split="train",
        pre_transform=lambda data: zinc_pre_transform(data, transform),
    )  # TODO:change subset=True before merge
    valset = ZINC(
        root="dataset/zinc",
        subset=False,
        split="val",
        pre_transform=lambda data: zinc_pre_transform(data, transform),
    )  # TODO:change subset=True before merge
    testset = ZINC(
        root="dataset/zinc",
        subset=False,
        split="test",
        pre_transform=lambda data: zinc_pre_transform(data, transform),
    )  # TODO:change subset=True before merge

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
        compute_grad_energy=args.compute_grad_energy,
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("Unexpected error:")
        log(sys.exc_info())
        log(e)
    sys.exit(0)
