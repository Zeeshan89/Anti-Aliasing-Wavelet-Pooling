#!/bin/bash
# Shared Frontier environment (sourced by pack8.sbatch). Mirrors Run_gpu_frontier_separate_mult.sh.
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env

export PROJECT_DIR=/lustre/orion/proj-shared/cli138/z7a/Project
export TAR_DIR=$PROJECT_DIR/data_tars           # one <dataset>.tar per dataset (make_tar.sh)
export NVME_DATA=/mnt/bb/$USER/data
export MIOPEN_ROOT=$PROJECT_DIR/miopen_cache     # per-GCD, on Lustre so it persists across chained jobs
export YOLO_OFFLINE=1                            # compute nodes have no internet
export MIOPEN_DISABLE_CACHE=0 MIOPEN_FIND_MODE=2 MIOPEN_LOG_LEVEL=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=7

# stage_dataset <name>: untar to NVMe once per job; echo the data.yaml to use.
stage_dataset() {
    local ds=$1 tar=$TAR_DIR/$1.tar dst=$NVME_DATA/$1
    if [[ ! -f $tar ]]; then            # no tar -> train straight from Lustre
        echo "$PROJECT_DIR/dataset/$ds/split/data.yaml"; return
    fi
    if [[ ! -f $dst/.staged ]]; then
        mkdir -p "$dst" && tar -xf "$tar" -C "$dst" \
            && sed -i "s#^path:.*#path: $dst#" "$dst/data.yaml" && touch "$dst/.staged"
    fi
    echo "$dst/data.yaml"
}
