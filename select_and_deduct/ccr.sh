#!/bin/sh
#
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name="reasoning"
#SBATCH --output=log2.out
#SBATCH --partition=rohini
#SBATCH --qos=rohini
#SBATCH --account=rohini
#SBATCH --cluster=faculty
#SBATCH --gpus-per-node=2

echo "submitting job..."
bash run_train_selector_llama.sh
echo "job finished successfully."
