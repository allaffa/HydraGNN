#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 16

NN=$SLURM_JOB_NUM_NODES
source /lustre/orion/cph161/world-shared/mlupopa/module-to-load-andes.sh
source /lustre/orion/cph161/world-shared/mlupopa/max_conda_envs_andes/bin/activate
conda activate hydragnn

export PYTHONPATH=/lustre/orion/cph161/world-shared/mlupopa/ADIOS_andes/install/lib/python3.9/site-packages:$PYTHONPATH

which python
#export PYTHONPATH=/dir/to/HydraGNN:$PYTHONPATH
export PYTHONPATH=/lustre/orion/cph161/proj-shared/zhangp/HydraGNN_EL:$PYTHONPATH


srun -n$((SLURM_JOB_NUM_NODES*4)) -c1 python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_lattice --dataname=alloy_binary_lattice --log="EL_lattice_andes"

srun -n$((SLURM_JOB_NUM_NODES*4)) -c1 python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_rmsd_1,examples/ensemble_learning/alloy_binary_rmsd_2  --dataname=alloy_binary_rmsd --log="EL_rmsd_andes"

srun -n$((SLURM_JOB_NUM_NODES*4)) -c1 python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_energy --log="EL_energy_andes"
