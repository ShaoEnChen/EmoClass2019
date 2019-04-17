#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE
#SBATCH --get-user-env=L

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=mlfinal
#SBATCH --time=47:59:59
#SBATCH --ntasks=28
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

#First Executable Line
module load Anaconda/3-5.0.0.1
source activate mlfinal
cd /scratch/user/seanchen47/EmoClass2019_sharpen/data
python gen_preprocessed.py --sharpen=True
cd ..
python main.py --bs=64 --bs-vt=64 --epoch=50
