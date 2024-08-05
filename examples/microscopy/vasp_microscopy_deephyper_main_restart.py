import os, sys

import torch

torch.backends.cudnn.enabled = False

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import pandas as pd
import subprocess
import re

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None

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

def read_results_from_csv(file_path: str) -> pd.DataFrame:
    """Read the results of a Hyperparameter Search from a CSV file.

    Args:
        file_path (str): the path to the CSV file.

    Returns:
        pd.DataFrame: the results of a Hyperparameter Search.
    """
    return pd.read_csv(file_path, header=0)


def _parse_results(stdout):
    pattern = r"Val Loss: ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
    matches = re.findall(pattern, stdout.decode())
    if matches:
        return matches[-1][0]
    else:
        return "F"


def run(trial, dequed=None):
    f = open(f"output-{trial.id}.txt", "w")
    python_exe = sys.executable
    python_script = os.path.join(os.path.dirname(__file__), "vasp_microscopy_deephyper_trial.py")

    # TODO: Launch a subprocess with `srun` to train neural networks
    params = trial.parameters
    log_name = "MO2" + "_" + str(trial.id)
    master_addr = f"HYDRAGNN_MASTER_ADDR={dequed[0]}"
    nodelist = ",".join(dequed)

    # time srun -u -n32 -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest
    prefix = " ".join(
        [
            f"srun -u",
            f"-N {NNODES_PER_TRIAL} -n {NGPUS_PER_TRIAL}",
            f"--ntasks-per-node=8 --gpus-per-node=8",
            f"--cpus-per-task {OMP_NUM_THREADS} --threads-per-core 1 --cpu-bind threads",
            f"--gpus-per-task=1 --gpu-bind=closest",
            f"--export=ALL,{master_addr},HYDRAGNN_MAX_NUM_BATCH=100,HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1,HYDRAGNN_AGGR_BACKEND=mpi",
            f"--nodelist={nodelist}",
            f"--output {DEEPHYPER_LOG_DIR}/output_{SLURM_JOB_ID}_{trial.id}.txt",
            f"--error {DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt",
        ]
    )

    command = " ".join(
        [
            prefix,
            python_exe,
            "-u",
            python_script,
            f"--model_type={trial.parameters['model_type']}",
            f"--hidden_dim={trial.parameters['hidden_dim']}",
            f"--num_conv_layers={trial.parameters['num_conv_layers']}",
            f"--num_sharedlayers={trial.parameters['num_sharedlayers']}",
            f"--dim_sharedlayers={trial.parameters['dim_sharedlayers']}",
            f"--num_headlayers={trial.parameters['num_headlayers']}",
            f"--dim_headlayers_graph={trial.parameters['dim_headlayers_graph']}",
            f"--dim_headlayers_node={trial.parameters['dim_headlayers_node']}",
            f"--pickle",
            ##f"--ddstore",
            ## debugging
            ##f'--multi_model_list="ANI1x"',
            f"--num_epoch=300",
            f"--log={log_name}",
        ]
    )
    print("Command = ", command, flush=True, file=f)

    output = "F"
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        pattern = r"Val Loss: ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
        fout = open(f"{DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt", "r")
        while True:
            line = fout.readline()
            matches = re.findall(pattern, line)
            if matches:
                output = -float(matches[-1][0])
            if not line:
                break
        fout.close()

    except Exception as excp:
        print(excp, flush=True, file=f)
        output = "F"

    print("Output:", output, flush=True, file=f)
    objective = output
    print(objective, flush=True, file=f)
    metadata = {"some_info": "some_value"}
    f.close()

    return {"objective": objective, "metadata": metadata}


if __name__ == "__main__":

    log_name = "MO2"

    # Choose the sampler (e.g., TPESampler or RandomSampler)
    from deephyper.evaluator import Evaluator, ProcessPoolEvaluator, queued
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from hydragnn.utils.deephyper import read_node_list
    #from deephyper.analysis.hps._hps import read_results_from_csv

    # define the variable you want to optimize
    problem = HpProblem()

    # Define the search space for hyperparameters
    ## suggestion: add learning rate as additional hyperparameter, and use log-uniform sampling
    ## add choice of optimizer as hyperparameter
    ## choice of activation function (ReLy, LeakyReLU, sigmoid)
    problem.add_hyperparameter((1, 5), "num_conv_layers")  # discrete parameter
    problem.add_hyperparameter((300, 2000), "hidden_dim")  # discrete parameter
    #problem.add_hyperparameter((300, 2000, "log-uniform"), "hidden_dim")  # discrete parameter, sample uniformly in log-scale
    problem.add_hyperparameter((1, 5), "num_sharedlayers")  # discrete parameter
    problem.add_hyperparameter((500, 2000), "dim_sharedlayers")  # discrete parameter
    problem.add_hyperparameter((1, 3), "num_headlayers")  # discrete parameter
    problem.add_hyperparameter((500, 1000), "dim_headlayers_graph")  # discrete parameter
    problem.add_hyperparameter((500, 3000), "dim_headlayers_node")  # discrete parameter
    problem.add_hyperparameter(
        ["EGNN", "SchNet", "PNA"], "model_type"
    )  # categorical parameter

    # Create the node queue
    queue, _ = read_node_list()
    print("The queue:", queue, len(queue))
    print("NNODES_PER_TRIAL", NNODES_PER_TRIAL)
    print("NUM_CONCURRENT_TRIALS", NUM_CONCURRENT_TRIALS)
    print("NGPUS_PER_TRIAL", NGPUS_PER_TRIAL)
    print("NTOTGPUS", NTOTGPUS)
    print(NTOTGPUS, NGPUS_PER_TRIAL, NTOTGPUS // NGPUS_PER_TRIAL, len(queue))

    # Define the search space for hyperparameters
    # define the evaluator to distribute the computation
    evaluator = queued(ProcessPoolEvaluator)(
        run,
        num_workers=NUM_CONCURRENT_TRIALS,
        queue=queue,
        queue_pop_per_task=NNODES_PER_TRIAL,  # Remove the hard-coded value later
    )

    # Define the search method and scalarization
    # search = CBO(problem, parallel_evaluator, random_state=42, log_dir=log_name)
    search = CBO(
        problem,
        evaluator,
        acq_func="UCBd", #UCB measures both aleatoric and epistemic uncertainty of performance, UCBd disregards aleatoric and works better than UCB
        multi_point_strategy="cl_min",  # Constant liar strategy, cl_max can help weigh more on exploitation and explore less
        random_state=42,
        # Location where to store the results
        log_dir=log_name,
        # Number of threads used to update surrogate model of BO
        n_jobs=OMP_NUM_THREADS,
    )
    #acq_optimizer="mixedga": decides how you optimize the funciton that selectes the next candidate. Default: random sampling of 10,000 candidate from your problem. When samples are fewer (expensive), use "mixedga"  
    # acq_optimizer_freq=1, recommended!
    # to force baseline to be evaluated by HPO, you can pass the baseline configuration to fit-surrogate, search.fit_surrogate("results.csv") - this i s NOT a bias, and you will continue samplign in a uniform way, to be used if the problem is the same

    print("MASSI: about to read existing results from csv")
    preloaded_results = read_results_from_csv("./preloaded_results.csv")

    print("MASSI: about to fit surrogate model")
    search.fit_surrogate(preloaded_results)
    #search_tl.fit_generative_model(preloaded_results) # replaces uniform sampling with biased sampling to generate new trials in regions of best performance, CAVEAT: USE IT ONLY FOR TRANSFER LEARNING (if you change dataset and/or problem)  

    timeout = None
    print("MASSI: about to start HPO search")
    results = search.search(max_evals=10000, timeout=timeout)
    print(results)

    sys.exit(0)
