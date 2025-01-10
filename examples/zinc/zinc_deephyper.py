import os, sys, json
import logging

import torch
import torch_geometric
from torch_geometric.datasets import ZINC
from torch_geometric.transforms import AddLaplacianEigenvectorPE

# torch.backends.cudnn.enabled = False # #TODO:change to 0.25 before merge

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn

def zinc_pre_transform(data, transform):
    data.x = data.x.float().view(-1, 1)
    data.edge_attr = data.edge_attr.float().view(-1, 1)
    data = transform(data)
    # gps requires relative edge features, introduced rel_lapPe as edge encodings
    source_pe = data.pe[data.edge_index[0]]
    target_pe = data.pe[data.edge_index[1]]
    data.rel_pe = torch.abs(source_pe - target_pe)  # Compute feature-wise difference
    return data

log_name = "zinc_hpo_trials"

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
train = ZINC(
    root="dataset/zinc", subset=False, split="train", pre_transform=lambda data: zinc_pre_transform(data, transform)
)
val = ZINC(
    root="dataset/zinc", subset=False, split="val", pre_transform=lambda data: zinc_pre_transform(data, transform)
)
test = ZINC(
    root="dataset/zinc", subset=False, split="test", pre_transform=lambda data: zinc_pre_transform(data, transform)
)

(train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
    train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
)

def run(trial):

    global config

    trial_config = config

    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()

    trial_log_name = log_name + "_" + str(trial.id)
    hydragnn.utils.print.print_utils.setup_log(trial_log_name)
    writer = hydragnn.utils.model.model.get_summary_writer(trial_log_name)

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    # log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    if trial.parameters["global_attn_heads"] is not None:
        trial_config["NeuralNetwork"]["Architecture"][
            "global_attn_heads"
        ] = trial.parameters["global_attn_heads"]
        global_attn_heads = trial.parameters["global_attn_heads"]
        hidden_dim = global_attn_heads * trial.parameters["hidden_dim"]
    else:
        hidden_dim = trial.parameters["hidden_dim"]

    # Update the config dictionary with the suggested hyperparameters
    trial_config["NeuralNetwork"]["Architecture"]["mpnn_type"] = trial.parameters[
        "mpnn_type"
    ]
    trial_config["NeuralNetwork"]["Architecture"]["hidden_dim"] = hidden_dim
    trial_config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = trial.parameters[
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

    if trial.parameters["mpnn_type"] not in ["EGNN", "DimeNet"]:
        trial_config["NeuralNetwork"]["Architecture"]["equivariance"] = False

    trial_config = hydragnn.utils.input_config_parsing.update_config(trial_config, train_loader, val_loader, test_loader)

    hydragnn.utils.input_config_parsing.save_config(trial_config, trial_log_name)

    model = hydragnn.models.create_model_config(
        config=trial_config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    learning_rate = trial_config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.model.model.load_existing_model_config(
        model, trial_config["NeuralNetwork"]["Training"], optimizer=optimizer
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
        trial_config["NeuralNetwork"],
        trial_log_name,
        verbosity,
        create_plots=False,
    )

    hydragnn.utils.model.model.save_model(model, optimizer, trial_log_name)
    hydragnn.utils.print.print_timers(verbosity)

    # Return the metric to minimize (e.g., validation loss)
    validation_loss, tasks_loss = hydragnn.train.validate(
        val_loader, model, verbosity, reduce_ranks=True
    )

    # Move validation_loss to the CPU and convert to NumPy object
    validation_loss = validation_loss.cpu().detach().numpy()

    # Return the metric to minimize (e.g., validation loss)
    # By default, DeepHyper maximized the objective function, so we need to flip the sign of the validation loss function
    print("validation_loss.item()", validation_loss.item())
    return -validation_loss.item()


if __name__ == "__main__":
    # Choose the sampler (e.g., TPESampler or RandomSampler)
    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import Evaluator

    # define the variable you want to optimize
    problem = HpProblem()

    # Define the search space for hyperparameters
    problem.add_hyperparameter((1, 4), "num_conv_layers")  # discrete parameter
    problem.add_hyperparameter((1,100), "hidden_dim")  # discrete parameter
    problem.add_hyperparameter((1, 3), "num_headlayers")  # discrete parameter
    problem.add_hyperparameter((1, 3), "dim_headlayers")  # discrete parameter
    
    # Include "global_attn_heads" to list of hyperparameters if global attention engine is used
    if config["NeuralNetwork"]["Architecture"]["global_attn_engine"] is not None:
        problem.add_hyperparameter([2, 4, 8], "global_attn_heads")  # discrete parameter
    problem.add_hyperparameter(
        ["EGNN", "PNA"], "mpnn_type"
    )  # categorical parameter

    # Define the search space for hyperparameters
    # define the evaluator to distribute the computation
    parallel_evaluator = Evaluator.create(
        run,
        method="process",
        method_kwargs={
            "num_workers": 1,
        },
    )

    # define your search and execute it
    search = CBO(problem, parallel_evaluator, random_state=42, log_dir=log_name)

    timeout = 1200
    results = search.search(max_evals=10, timeout=timeout)
    print(results)

    sys.exit(0)
