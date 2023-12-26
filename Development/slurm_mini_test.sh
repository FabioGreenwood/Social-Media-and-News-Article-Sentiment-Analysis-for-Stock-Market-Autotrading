#!/bin/bash
#SBATCH --job-name=FG_mini_test
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:30:00

module load python  # Load the Python module if needed

python /home/accounts/studenti/id110xaq/final_project/Development/log.txt /home/accounts/studenti/id110xaq/final_project/Development/experiment_handler_mini.py