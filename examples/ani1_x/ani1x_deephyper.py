import os
import sys
import json
import subprocess
import re

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

# Load config for reference (will be passed to training script)
DIRPWD = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(DIRPWD, "ani1x_energy.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Set dataset paths
MODELNAME = "ANI1x"

def _parse_results(stdout):
    """Parse validation loss from training script output."""
    pattern = r"Train Loss: ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
    matches = re.findall(pattern, stdout.decode())
    # By default, DeepHyper maximizes the objective function, so we need to flip the sign
    if matches:
        return -float(matches[-1][0])
    else:
        return "F"


def run(trial, dequed=None):
    """Run a single trial using subprocess isolation."""
    f = open(f"output-{trial.id}.txt", "w")
    python_exe = sys.executable
    python_script = os.path.join(os.path.dirname(__file__), "train_hpo.py")

    # Get trial hyperparameters
    params = trial.parameters
    log_name = f"ani1x_{trial.id}"
    
    # Set master address from dequeued node
    master_addr = f"HYDRAGNN_MASTER_ADDR={dequed[0]}"
    nodelist = ",".join(dequed)

    # Build srun command for distributed training
    prefix = " ".join(
        [
            f"srun",
            f"-N {NNODES_PER_TRIAL} -n {NGPUS_PER_TRIAL}",
            f"--ntasks-per-node=8 --gpus-per-node=8",
            f"--cpus-per-task {OMP_NUM_THREADS} --threads-per-core 1 --cpu-bind threads",
            f"--gpus-per-task=1 --gpu-bind=closest",
            f"--export=ALL,{master_addr}",
            f"--nodelist={nodelist}",
            f"--output {DEEPHYPER_LOG_DIR}/output_{SLURM_JOB_ID}_{trial.id}.txt",
            f"--error {DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt",
        ]
    )

    # Build full command with hyperparameters
    command = " ".join(
        [
            prefix,
            python_exe,
            "-u",
            python_script,
            f"--inputfile=ani1x_energy.json",
            f"--modelname={MODELNAME}",
            f"--log={log_name}",
            f"--mpnn_type={params['mpnn_type']}",
            f"--hidden_dim={params['hidden_dim']}",
            f"--num_conv_layers={params['num_conv_layers']}",
            f"--num_headlayers={params['num_headlayers']}",
            f"--dim_headlayers={params['dim_headlayers']}",
        ]
    )
    
    print("Command = ", command, flush=True, file=f)

    output = "F"
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        output = _parse_results(result)
    except Exception as excp:
        print(excp, flush=True, file=f)
        output = "F"

    print("Got the output", output, flush=True, file=f)
    objective = output
    print(objective, flush=True, file=f)
    metadata = {"trial_id": trial.id}
    f.close()

    return {"objective": objective, "metadata": metadata}

if __name__ == "__main__":

    log_name = "ani1x_deephyper"

    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import Evaluator, ProcessPoolEvaluator, queued
    from hydragnn.utils.hpo.deephyper import read_node_list

    problem = HpProblem()
    problem.add_hyperparameter((1, 4), "num_conv_layers")
    problem.add_hyperparameter((32, 512), "hidden_dim")
    problem.add_hyperparameter((1, 3), "num_headlayers")
    problem.add_hyperparameter((32, 256), "dim_headlayers")
    problem.add_hyperparameter(["EGNN", "PNA", "SchNet", "DimeNet", "MACE"], "mpnn_type")

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
    search = CBO(
        problem,
        evaluator,
        acq_func="UCB",
        multi_point_strategy="cl_min",  # Constant liar strategy
        # random_state=42,
        # Location where to store the results
        log_dir=log_name,
        # Number of threads used to update surrogate model of BO
        n_jobs=OMP_NUM_THREADS,
    )

    timeout = None
    results = search.search(max_evals=200, timeout=timeout)

    print(results)
    sys.exit(0)
