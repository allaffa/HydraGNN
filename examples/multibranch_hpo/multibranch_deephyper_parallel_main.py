import os, sys, json
import logging

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
import time
import argparse

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None

# Retrieve constants
NNODES = int(os.environ["NNODES"])
NPROCS_PER_NODE=int(os.environ["NPROCS_PER_NODE"])
NDEPTH=int(os.environ["NDEPTH"])
NTOTGPUS = int(os.environ["NTOTGPUS"])
NNODES_PER_TRIAL = int(os.environ["NNODES_PER_TRIAL"])
NGPUS_PER_TRIAL = int(os.environ["NGPUS_PER_TRIAL"])
NUM_CONCURRENT_TRIALS = int(os.environ["NUM_CONCURRENT_TRIALS"])
NTOT_DEEPHYPER_RANKS = int(os.environ["NTOT_DEEPHYPER_RANKS"])
OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
DEEPHYPER_LOG_DIR = os.environ["DEEPHYPER_LOG_DIR"]
DEEPHYPER_DB_HOST = os.environ["DEEPHYPER_DB_HOST"]
PBS_JOBID = os.environ["PBS_JOBID"]

# Retrieve HydraGNN constants
HYDRAGNN_TRACE_LEVEL = int(os.environ["HYDRAGNN_TRACE_LEVEL"])
HYDRAGNN_MAX_NUM_BATCH = int(os.environ["HYDRAGNN_MAX_NUM_BATCH"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
NUM_EPOCHS = int(os.environ["NUM_EPOCHS"])

# Retrieve Aurora envs
CCL_KVS_MODE = os.environ.get("CCL_KVS_MODE", "mpi")
CCL_KVS_CONNECTION_TIMEOUT = os.environ.get("CCL_KVS_CONNECTION_TIMEOUT", "300")
FI_MR_CACHE_MONITOR = os.environ.get("FI_MR_CACHE_MONITOR", "userfaultfd")


os.makedirs(DEEPHYPER_LOG_DIR)

def _parse_results(stdout):
    pattern = r"Val Loss: ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
    matches = re.findall(pattern, stdout.decode())
    if matches:
        return matches[-1][0]
    else:
        return "F"


def run(trial, dequed=None):
    f = open(f"{DEEPHYPER_LOG_DIR}/output-{trial.id}.txt", "w")
    print("dequed = ", dequed, flush=True, file=f)
    python_exe = sys.executable
    python_script = os.path.join(
        os.path.dirname(__file__), "train.py"
    )

    # TODO: Launch a subprocess with `srun` to train neural networks
    params = trial.parameters
    log_name = f"multibranch_hpo_trials_{PBS_JOBID}_{str(trial.id)}"

    prefix = " ".join(
        [
            f"mpiexec", 
            f"-n {NGPUS_PER_TRIAL}",
            f"--ppn {NPROCS_PER_NODE}", 
            f"--host {','.join(dequed)}",
            f"--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100", 
            ## HydraGNN envs
            f"--env HYDRAGNN_TRACE_LEVEL={HYDRAGNN_TRACE_LEVEL}",
            f"--env HYDRAGNN_MAX_NUM_BATCH={HYDRAGNN_MAX_NUM_BATCH}",
            # ## Aurora envs
            # f"--env CCL_KVS_MODE={CCL_KVS_MODE}",
            # f"--env CCL_KVS_CONNECTION_TIMEOUT={CCL_KVS_CONNECTION_TIMEOUT}",
            # f"--env FI_MR_CACHE_MONITOR={FI_MR_CACHE_MONITOR}",
        ]
    )

    ## command example:
    # python -u ./examples/multibranch/train.py --log=GFM_taskparallel_weak-$PBS_JOBID-NN$NNODES-BS$BATCH_SIZE-TP1-DD$HYDRAGNN_DDSTORE_METHOD-NW$HYDRAGNN_NUM_WORKERS --everyone \
    # --inputfile=multibranch_GFM260.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
    # --multi --ddstore --multi_model_list=$datadir5 --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCHS \
    # --task_parallel --use_devicemesh --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH))


    command = " ".join(
        [
            prefix,
            python_exe,
            "-u",
            python_script,
            f"--inputfile=multibranch_GFM260.json",
            f"--mpnn_type={trial.parameters['mpnn_type']}",
            f"--hidden_dim={trial.parameters['hidden_dim']}",
            f"--num_conv_layers={trial.parameters['num_conv_layers']}",
            f"--num_headlayers={trial.parameters['num_headlayers']}",
            f"--dim_headlayers={trial.parameters['dim_headlayers']}",
            f"--num_samples={BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH*2}",
            f"--multi",
            f"--ddstore",
            f"--multi_model_list=ANI1x,qm7x,MPTrj,Alexandria,transition1x,OC2020_all,OC2022,OMat24,OMol25",
            f"--batch_size={BATCH_SIZE}",
            f"--num_epoch={NUM_EPOCHS}",
            f"--task_parallel",
            f"--use_devicemesh",
            f"--oversampling",
            f"--oversampling_num_samples={BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH}",
            f"--log={log_name}",
            f"--everyone",
            ## stdout/stderr
            f"> {DEEPHYPER_LOG_DIR}/output_{PBS_JOBID}_{trial.id}.txt",
            f"2> {DEEPHYPER_LOG_DIR}/error_{PBS_JOBID}_{trial.id}.txt",
        ]
    )
    print("Command = ", command, flush=True, file=f)

    output = "F"
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        pattern = r"Val Loss: ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
        fout = open(f"{DEEPHYPER_LOG_DIR}/error_{PBS_JOBID}_{trial.id}.txt", "r")
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--mpnn_type", required=True, help="list of mpnn types separated by comma") ## ["EGNN", "SchNet", "PNAEq", "MACE", "PAINN"]
    args = parser.parse_args()    

    log_name = f"multibranch_hpo_trials-{'-'.join(args.mpnn_type.split(','))}"

    # Choose the sampler (e.g., TPESampler or RandomSampler)
    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import Evaluator, ProcessPoolEvaluator, queued
    from hydragnn.utils.hpo.deephyper import read_node_list_pbs

    # define the variable you want to optimize
    problem = HpProblem()

    # Define the search space for hyperparameters
    problem.add_hyperparameter((2, 6), "num_conv_layers")  # discrete parameter
    problem.add_hyperparameter((1000, 5000), "hidden_dim")  # discrete parameter
    problem.add_hyperparameter((2, 3), "num_headlayers")  # discrete parameter
    problem.add_hyperparameter((500, 2000), "dim_headlayers")  # discrete parameter
    problem.add_hyperparameter(args.mpnn_type.split(","), "mpnn_type") # categorical parameter

    # Create the node queue
    queue, _ = read_node_list_pbs()
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
        acq_func="UCB",
        multi_point_strategy="cl_min",  # Constant liar strategy
        random_state=42,
        # Location where to store the results
        log_dir=log_name,
        # Number of threads used to update surrogate model of BO
        n_jobs=OMP_NUM_THREADS,
    )

    fname = os.path.join(log_name, "preloaded_results.csv")
    print("Try to preload:", fname)
    if os.path.exists(fname):
        t0 = time.time()
        print("Read existing results:", fname)
        preloaded_results = pd.read_csv(fname, header=0)
        search.fit_surrogate(preloaded_results)
        t1 = time.time()
        print("Fit done:", t1-t0)    

    timeout = None
    results = search.search(max_evals=100, timeout=timeout)
    print(results)

    sys.exit(0)
