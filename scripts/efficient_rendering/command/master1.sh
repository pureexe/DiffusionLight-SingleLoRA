#!/bin/sh

#SBATCH --error=output/jobs/err/master1.%j  
#SBATCH --output=output/jobs/out/master1.%j
#SBATCH --job-name=rMaster1       # Job name
#SBATCH --mem=256GB                  # Memory request for this job
#SBATCH --nodes=1                   # The number of nodes
#SBATCH --partition=cpu
#SBATCH --account=vision
#SBATCH --time=72:0:0                # Runing time 2 days
#SBATCH --cpus-per-task=80

SINGULARITYENV_HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/
SINGULARITYENV_HUB_HOME=/ist/ist-share/vision/huggingface/

singularity exec --bind /ist:/ist  --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets /ist/ist-share/vision/pakkapon/singularity/diffusers0310v6.sif python efficient_sh.py --total 2 --idx 0 --threads 40
