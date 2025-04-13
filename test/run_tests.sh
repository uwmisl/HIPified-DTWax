#!/bin/bash
#SBATCH --job-name=testing_dtwax   # Job name
#SBATCH --output=logs/%x_%j.out    # Stdout log (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err     # Stderr log
#SBATCH -N 1                       # Total number of nodes requested
#SBATCH -n 1                       # Total number of mpi tasks requested
#SBATCH -t 00:30:00                # Run time (hh:mm:ss)
#SBATCH -p devel                   # Desired partition

source ~/py3_env/bin/activate
python run_tests.py