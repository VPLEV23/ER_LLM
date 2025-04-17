#!/bin/bash -l
#SBATCH -p cuda                 # Use the cuda partition
#SBATCH -N 1                    # Request 1 node
#SBATCH -n 1                    # Request 1 task
#SBATCH -J MyJobName            # Set the job name
#SBATCH -c 9                  # Request 25 cores
#SBATCH --gres=gpu:large:1        # Request 1 instance of the 'large' GPU type

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
conda activate env
srun python "$@"