#!/bin/bash
#SBATCH --job-name=cache
#SBATCH --time=20:30:00
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/cache_run%j.out
#SBATCH --error=slurm/cache_run%j.out

# Load required modules
module load apptainer
module load cuda/cuda-12.4.0
cd ~/git/activation-cache-exporter

# Run all commands inside the Apptainer container
apptainer exec --nv ~/containers/train-transformer_latest.sif bash <<zzzRunHEREinTheContainer
nvidia-smi
python prepare.py --output_dir data/activations/sorted \
    --sort_ds_by_len \
    --batch_size 128  \
    --auto_find_batch_size \
    --batches_per_shard 1 \
    --max_shards_created 480 \
    --model_checkpoint austindavis/chessGPT2 \
    --ds_config 202302-00000-00009 \
    --ds_repo austindavis/lichess-uci-scored \
    --ds_split train \
    --ds_input_column Transcript \
    --ds_label_columns Site WhiteElo BlackElo Transcript Scores \
    --n_pos 1024 \
    --log_file log.txt
exit
zzzRunHEREinTheContainer

scancel "$SLURM_JOB_ID"