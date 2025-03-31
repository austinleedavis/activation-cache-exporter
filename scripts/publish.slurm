#!/bin/bash
#SBATCH --job-name=publish
#SBATCH --time=20:30:00
#SBATCH --partition=highgpu
#SBATCH --cpus-per-task=100
#SBATCH --output=slurm/publish_run%j.out
#SBATCH --error=slurm/publish_run%j.out

# Load required modules
module load apptainer
module load cuda/cuda-12.4.0
cd ~/git/activation-cache-exporter

# Run all commands inside the Apptainer container
apptainer exec ~/containers/train-transformer_latest.sif bash <<zzzRunHEREinTheContainer
nvidia-smi
pip install -r requirements.txt
HF_HUB_ENABLE_HF_TRANSFER=1 python publish.sh data/activations/sorted austindavis/chessGPT2_hidden_states
exit
zzzRunHEREinTheContainer

scancel "$SLURM_JOB_ID"