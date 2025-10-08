#!/bin/bash

#SBATCH --job-name=cog_app_llama8        
#SBATCH --output=qwen7.txt           
#SBATCH --partition=nmes_gpu
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1
#SBATCH --mem 40GB
#SBATCH --constraint a100
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=4             
#SBATCH --time=2-00                
#SBATCH --mail-type=ALL               
#SBATCH --mail-user=hainiu.xu@kcl.ac.uk    

python exp.py --model qwen7
