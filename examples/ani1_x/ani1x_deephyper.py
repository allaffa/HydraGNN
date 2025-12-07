import os
import sys
import json
import logging
import torch
from mpi4py import MPI
import hydragnn
from hydragnn.utils.print.print_utils import log
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.adiosdataset import AdiosDataset
from hydragnn.preprocess import create_dataloaders
from hydragnn.utils.input_config_parsing import update_config, save_config
from hydragnn.train import train_validate_test, validate

# Fix random seed for reproducibility
random_state = 0
torch.manual_seed(random_state)

# Retrieve constants
NNODES = int(os.environ["NNODES"])
NTOTGPUS = int(os.environ["NTOTGPUS"])
NNODES_PER_TRIAL = int(os.environ["NNODES_PER_TRIAL"])
NGPUS_PER_TRIAL = int(os.environ["NGPUS_PER_TRIAL"])
NUM_CONCURRENT_TRIALS = int(os.environ["NUM_CONCURRENT_TRIALS"])
NTOT_DEEPHYPER_RANKS = int(os.environ["NTOT_DEEPHYPER_RANKS"])
OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
DEEPHYPER_LOG_DIR = os.environ["DEEPHYPER_LOG_DIR"]
DEEPHYPER_DB_HOST = os.environ["DEEPHYPER_DB_HOST"]
SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]

# Load config
DIRPWD = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(DIRPWD, "ani1x_energy.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]

# Set dataset paths
MODELNAME = "ANI1x"
BP_PATH = os.path.join(DIRPWD, "dataset", f"{MODELNAME}.bp")

# Variable config
var_config = config["NeuralNetwork"]["Variables_of_interest"]
var_config["graph_feature_names"] = ["energy"]
var_config["graph_feature_dims"] = [1]
var_config["node_feature_names"] = ["atomic_number", "cartesian_coordinates", "forces"]
var_config["node_feature_dims"] = [1, 3, 3]

# Load ADIOS datasets
comm = MPI.COMM_WORLD
opt = {"preload": False, "shmem": False, "ddstore": False, "ddstore_width": None}
trainset = AdiosDataset(BP_PATH, "trainset", comm, **opt, var_config=var_config)
valset = AdiosDataset(BP_PATH, "valset", comm, **opt, var_config=var_config)
testset = AdiosDataset(BP_PATH, "testset", comm, **opt, var_config=var_config)

(train_loader, val_loader, test_loader) = create_dataloaders(
    trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
)

def run(trial):
    global config
    trial_config = json.loads(json.dumps(config))  # Deep copy
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    trial_log_name = f"ani1x_{trial.id}"
    hydragnn.utils.print.print_utils.setup_log(trial_log_name)
    writer = hydragnn.utils.model.get_summary_writer(trial_log_name)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s (rank {rank}): %(message)s",
        datefmt="%H:%M:%S",
    )
    # Update config with trial hyperparameters
    trial_config["NeuralNetwork"]["Architecture"]["mpnn_type"] = trial.parameters["mpnn_type"]
    trial_config["NeuralNetwork"]["Architecture"]["hidden_dim"] = trial.parameters["hidden_dim"]
    trial_config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = trial.parameters["num_conv_layers"]
    trial_config["NeuralNetwork"]["Architecture"]["num_headlayers"] = trial.parameters["num_headlayers"]
    trial_config["NeuralNetwork"]["Architecture"]["dim_headlayers"] = trial.parameters["dim_headlayers"]
    # Update config with loaders
    trial_config = update_config(trial_config, train_loader, val_loader, test_loader)
    save_config(trial_config, trial_log_name)
    model = hydragnn.models.create_model_config(
        config=trial_config["NeuralNetwork"], verbosity=verbosity
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)
    learning_rate = trial_config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )
    hydragnn.utils.model.load_existing_model_config(
        model, trial_config["NeuralNetwork"]["Training"], optimizer=optimizer
    )
    train_validate_test(
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
    hydragnn.utils.model.save_model(model, optimizer, trial_log_name)
    if writer is not None:
        writer.close()
    validation_loss, _ = validate(val_loader, model, verbosity, reduce_ranks=True)
    validation_loss = validation_loss.cpu().detach().numpy()
    return -validation_loss.item()

if __name__ == "__main__":
    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import Evaluator, ProcessPoolEvaluator, queued
    problem = HpProblem()
    problem.add_hyperparameter((1, 4), "num_conv_layers")
    problem.add_hyperparameter((32, 512), "hidden_dim")
    problem.add_hyperparameter((1, 3), "num_headlayers")
    problem.add_hyperparameter((32, 256), "dim_headlayers")
    problem.add_hyperparameter(["EGNN", "PNA", "SchNet", "DimeNet", "MACE"], "mpnn_type")
    parallel_evaluator = Evaluator.create(
        run,
        method="process",
        acq_func="UCB",
        multi_point_strategy="cl_min",  # Constant liar strategy
        method_kwargs={"num_workers": 1},
        n_jobs=OMP_NUM_THREADS,
    )
    search = CBO(problem, parallel_evaluator, random_state=42, log_dir="ani1x_hpo")
    timeout = 1200
    results = search.search(max_evals=10, timeout=timeout)
    print(results)
    sys.exit(0)
