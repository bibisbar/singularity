#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=sim
# debug info

python sim_cal.py