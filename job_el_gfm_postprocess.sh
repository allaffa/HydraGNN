#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -N 1 
##SBATCH -S 1

module reset
module load PrgEnv-gnu
module load rocm/5.7.1
module load cmake
module load craype-accel-amd-gfx90a
module load amd-mixed/5.7.1
module load cray-mpich/8.1.26
module load miniforge3/23.11.0
module unload darshan-runtime

source activate /lustre/orion/world-shared/cph161/jyc/frontier/sw/envs/hydragnn-py39-rocm571-amd

## Use ADM build
PYTORCH_DIR=/autofs/nccs-svm1_sw/crusher/amdsw/karldev/pytorch-2.2.2-rocm5.7.1
PYG_DIR=/autofs/nccs-svm1_sw/crusher/amdsw/karldev/pyg-rocm5.7.1
export PYTHONPATH=$PYG_DIR:$PYTORCH_DIR:$PYTHONPATH

module use -a /lustre/orion/world-shared/cph161/jyc/frontier/sw/modulefiles
module load adios2/2.9.2-mpich-8.1.26

module load aws-ofi-rccl/devel-rocm5.7.1
echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1


## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

#export PYTHONPATH=/dir/to/HydraGNN:$PYTHONPATH
export PYTHONPATH=/lustre/orion/cph161/proj-shared/zhangp/HydraGNN_EL:$PYTHONPATH

#srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_GFM_bydataset_postprocess.py  --models_dir_folder=examples/ensemble_learning/GFM_logs --multi_model_list=ANI1x-v2,MPTrj-v2,OC2020-20M-v2,OC2022-v2,qm7x-v2

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_GFM_bydataset_postprocess.py \
--models_dir_folder=examples/ensemble_learning/GFM_logs_1028  --log=GFM_EnsembleInference_1028 --multi_model_list=ANI1x-v3,MPTrj-v3,OC2020-20M-v3,OC2022-v3,qm7x-v3

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_GFM_bydataset_postprocess.py \
--models_dir_folder=examples/ensemble_learning/GFM_logs_1028  --log=GFM_EnsembleInference_1028 --multi_model_list=ANI1x-v3

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_GFM_bydataset_postprocess.py \
--models_dir_folder=examples/ensemble_learning/GFM_logs_1028  --log=GFM_EnsembleInference_1028 --multi_model_list=MPTrj-v3

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_GFM_bydataset_postprocess.py \
--models_dir_folder=examples/ensemble_learning/GFM_logs_1028  --log=GFM_EnsembleInference_1028 --multi_model_list=OC2020-20M-v3

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_GFM_bydataset_postprocess.py \
--models_dir_folder=examples/ensemble_learning/GFM_logs_1028  --log=GFM_EnsembleInference_1028 --multi_model_list=OC2022-v3

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_GFM_bydataset_postprocess.py \
--models_dir_folder=examples/ensemble_learning/GFM_logs_1028  --log=GFM_EnsembleInference_1028 --multi_model_list=qm7x-v3

