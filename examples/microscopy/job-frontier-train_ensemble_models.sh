#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1

set -x
export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_AGGR_BACKEND=mpi

BASE_PORT=3442

source /lustre/orion/cph161/world-shared/mlupopa/module-to-load-frontier-rocm600.sh
source /lustre/orion/cph161/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn_rocm600

export PYTHONPATH=/lustre/orion/cph161/world-shared/mlupopa/ADIOS_frontier_rocm600/install/lib/python3.11/site-packages/:$PYTHONPATH

export MPLCONFIGDIR=/lustre/orion/cph161/world-shared/mlupopa/
export PYTHONPATH=$PWD:$PYTHONPATH

cd examples/microscopy/

#srun -n$((SLURM_JOB_NUM_NODES*8)) -c 1 python -u vasp_microscopy.py --inputfile vasp_multitasking.json --pickle
#srun -n$((SLURM_JOB_NUM_NODES)) -c 1 --gres=gpu:1 --gpus-per-task=1 --ntasks-per-gpu=1 --gpu-bind=closest python -u vasp_microscopy.py --inputfile third_deephyper_hpo_best_models_13.json --pickle --log third_deephyper_hpo_best_models_13

# Define the commands to be run
models=(
    third_deephyper_hpo_best_models_13
    third_deephyper_hpo_best_models_64
    third_deephyper_hpo_best_models_133
    #fourth_deephyper_hpo_best_models_9
    fourth_deephyper_hpo_best_models_154
    fourth_deephyper_hpo_best_models_176
    fourth_deephyper_hpo_best_models_181
    fourth_deephyper_hpo_best_models_190
    fourth_deephyper_hpo_best_models_192
)
nmodel=${#models[*]}
# Loop through the commands and run them in parallel
for((i=0; i<$nmodel; i++)); do
    model="${models[i]}"
    echo "Running: $model"
    HYDRAGNN_MASTER_PORT=$((BASE_PORT + i)) srun -n1 -c1 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest python -u vasp_microscopy.py --inputfile "$model.json" --pickle --log "$model" &
done

# Wait for all background jobs to finish
wait

echo "All commands have completed."
