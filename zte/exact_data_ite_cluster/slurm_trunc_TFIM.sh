#!/bin/bash -l
#SBATCH --job-name=jw_ite_tfim
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julia.gerecke@epfl.ch
#SBATCH --output=/home/guignon/QIC_project/pp-em/zte/exact_data_ite_cluster/output/%j.out
#SBATCH --error=/home/guignon/QIC_project/pp-em/zte/exact_data_ite_cluster/output/%j.err
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10 ##helvetios has 18 cores per node
#SBATCH --time=24:00:00

# Load necessary modules
module load gcc julia

echo "STARTING AT $(date)"

# Change to your project directory
# cd /home/gerecke/pp-em/zte/exact_data_ite_cluster

# Run your script using the Julia environment in ~/PauliProp
srun julia --project=/home/guignon/PP truncated_data_gen_TFIM.jl

echo "FINISHED AT $(date)"
