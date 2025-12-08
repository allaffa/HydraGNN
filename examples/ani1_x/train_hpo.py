"""
Training script for ANI1x with hyperparameter optimization support.
This script is designed to be called as a subprocess from ani1x_deephyper.py
"""
import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import random
import torch

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

import torch.distributed as dist

import hydragnn
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.adiosdataset import AdiosDataset
from hydragnn.preprocess import create_dataloaders
from hydragnn.utils.input_config_parsing import update_config, save_config
from hydragnn.train import train_validate_test, validate


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--inputfile", help="input file", type=str, required=True)
    parser.add_argument("--modelname", help="model name", type=str, default="ANI1x")
    parser.add_argument("--log", help="log name", type=str, required=True)
    
    # Hyperparameters from DeepHyper
    parser.add_argument("--mpnn_type", help="MPNN type", type=str, required=True)
    parser.add_argument("--hidden_dim", help="Hidden dimension", type=int, required=True)
    parser.add_argument("--num_conv_layers", help="Number of conv layers", type=int, required=True)
    parser.add_argument("--num_headlayers", help="Number of head layers", type=int, required=True)
    parser.add_argument("--dim_headlayers", help="Dimension of head layers", type=int, required=True)
    
    args = parser.parse_args()

    # Load configuration
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    
    with open(input_filename, "r") as f:
        config = json.load(f)
    
    verbosity = config["Verbosity"]["level"]
    
    # Update config with hyperparameters from command line
    config["NeuralNetwork"]["Architecture"]["mpnn_type"] = args.mpnn_type
    config["NeuralNetwork"]["Architecture"]["hidden_dim"] = args.hidden_dim
    config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = args.num_conv_layers
    config["NeuralNetwork"]["Architecture"]["num_headlayers"] = args.num_headlayers
    config["NeuralNetwork"]["Architecture"]["dim_headlayers"] = args.dim_headlayers
    
    # Variable config
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = ["energy"]
    var_config["graph_feature_dims"] = [1]
    var_config["node_feature_names"] = ["atomic_number", "cartesian_coordinates", "forces"]
    var_config["node_feature_dims"] = [1, 3, 3]
    
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    
    comm = MPI.COMM_WORLD
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s (rank {rank}): %(message)s",
        datefmt="%H:%M:%S",
    )
    
    log_name = args.log
    hydragnn.utils.print.print_utils.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)
    
    # Load ADIOS datasets with preload to avoid concurrent file access issues
    opt = {"preload": True, "shmem": False, "ddstore": False, "ddstore_width": None}
    fname = os.path.join(dirpwd, "dataset", f"{args.modelname}-v2.bp")
    
    trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
    valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
    testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
    
    info(
        f"trainset, valset, testset size: {len(trainset)} {len(valset)} {len(testset)}"
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    
    # Update config with loaders
    config = update_config(config, train_loader, val_loader, test_loader)
    save_config(config, log_name)
    
    # Create model
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"], verbosity=verbosity
    )
    
    # Setup optimizer and scheduler
    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )
    
    # Wrap model for distributed training
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)
    
    # Print model details
    print_model(model)
    
    # Load existing model if available
    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )
    
    # Train, validate, and test
    train_validate_test(
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
    
    # Save model
    hydragnn.utils.model.save_model(model, optimizer, log_name)
    
    if writer is not None:
        writer.close()
    
    # Validate and print final validation loss
    validation_loss, _ = validate(val_loader, model, verbosity, reduce_ranks=True)
    
    if rank == 0:
        validation_loss_val = validation_loss.cpu().detach().numpy()
        print(f"Train Loss: {validation_loss_val.item()}")
    
    dist.destroy_process_group()
    sys.exit(0)
