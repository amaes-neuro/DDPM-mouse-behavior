#!/bin/bash

#SBATCH --job-name=parallel_sampling
#SBATCH --output=out/sampling_%A_%a.out
#SBATCH --error=err/sampling_%A_%a.err
#SBATCH --array=0-999%20
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=4G

# variable declaration
model="t_4"
dataset_name="balanced4"

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# main task
python synthetic_probs_gen_par.py $model $dataset_name $SLURM_ARRAY_TASK_ID

# clean up
cd $SLURM_SUBMIT_DIR
