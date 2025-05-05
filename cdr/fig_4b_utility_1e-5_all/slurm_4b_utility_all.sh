#!/bin/bash -l
#SBATCH --job-name=utility_4b_1e-5_all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julia.gerecke@epfl.ch
#SBATCH --output=/home/gerecke/pp-em/cdr/fig_4b_utility_1e-5_all/output/%j.out
#SBATCH --error=/home/gerecke/pp-em/cdr/fig_4b_utility_1e-5_all/output/%j.err
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00

# Load necessary modules
module load gcc openmpi openblas julia

echo "STARTING AT $(date)"

# Change to your project directory
cd /home/gerecke/pp-em/cdr/fig_4b_utility_1e-5_all

# Run your script using the Julia environment in ~/PauliProp
srun julia --project=/home/gerecke/PauliProp IBM_utility_exp_4b_all.jl 

echo "FINISHED AT $(date)"