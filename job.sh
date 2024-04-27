#!/bin/bash
#SBATCH --job-name=AFNTankTrouble
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=40Gb
#SBATCH --gres=gpu:1

. $HOME/micromamba/etc/profile.d/micromamba.sh
micromamba activate "afn"
python afn_tanktrouble.py
