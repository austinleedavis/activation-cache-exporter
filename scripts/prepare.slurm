#!/bin/bash
#SBATCH --job-name=cache
#SBATCH --time=20:30:00
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm/prepare%j.out
#SBATCH --error=slurm/prepare%j.out

# Load required modules
module load apptainer
module load cuda/cuda-12.4.0

# Avg transcript len = 336 characters -> 205 pos/game -> 1024 records ~ 50MB shard size

# Run all commands inside the Apptainer container
apptainer exec --nv ~/containers/train-transformer_latest.sif bash <<zzzRunHEREinTheContainer
nvidia-smi
python prepare.py --output_dir data/activations/lichess-uci-201410 \
    --sort_ds_by_len \
    --batch_size 128  \
    --auto_find_batch_size \
    --sort_ds_reversed \
    --records_per_shard 1024 \
    --model_checkpoint austindavis/chessGPT2 \
    --ds_config 201410 \
    --ds_repo austindavis/lichess-uci \
    --ds_split train \
    --ds_input_column Transcript \
    --ds_label_columns Site WhiteElo BlackElo Transcript \
    --n_pos 1024 \
    --log_file log.txt
exit
zzzRunHEREinTheContainer

scancel "$SLURM_JOB_ID"