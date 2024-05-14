#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=sim

python /home/wiss/zhang/Jinhe/singularity/eccv_reb/ret_anet/calculate.py