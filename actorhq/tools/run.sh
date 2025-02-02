#!/bin/bash
#SBATCH --job-name=actorhq
#SBATCH --account=st-rhodin-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/st-rhodin-1/users/chunjin/project/data_processing/log/array_%A_%a.out
#SBATCH --error=/scratch/st-rhodin-1/users/chunjin/project/data_processing/log/array_%A_%a.err
#SBATCH --array=1-4

# load modules if not restore in your .bashrc
# Module load <package>
# Module load <package>

source activate v2a_env
#sockeye
cd /scratch/st-rhodin-1/users/chunjin/project/data_processing/data_process_4_humannerf/code/actorshq
id=${SLURM_ARRAY_TASK_ID}


subjects=('Actor08' 'Actor08' 'Actor07' 'Actor07')
actor=${subjects[id-1]}

sequences=('Sequence1' 'Sequence2' 'Sequence1' 'Sequence2')
sequence=${sequences[id-1]}


python download_manager.py \
    actor=${actor} \
    sequence=${sequence}
