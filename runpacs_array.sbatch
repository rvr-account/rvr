#!/bin/bash
# runpacs_array.sbatch
#
#SBATCH -J runpacs_array
#SBATCH -t 0-144:00
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu
#SBATCH --mem=50000
#SBATCH -o outfiles/slurm-%A-%a.out # STDOUT
#SBATCH -e outfiles/slurm-%A-%a.err # STDERR

module load Anaconda3/5.0.1-fasrc01
module load cuda/9.0-fasrc02 cudnn/7.4.1.5_cuda9.0-fasrc01

source activate tf1.12_cuda9

source sweeps/pacs_sweep/array_commands/${SLURM_ARRAY_TASK_ID}.sh
