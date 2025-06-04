#!/bin/sh

#SBATCH --error=output/jobs/err/004.%j  
#SBATCH --output=output/jobs/out/004.%j
#SBATCH --job-name=004       # Job name
#SBATCH --mem=32GB                  # Memory request for this job
#SBATCH --nodes=1                   # The number of nodes
#SBATCH --partition=gpu-4080
#SBATCH --account=vision
#SBATCH --time=72:0:0                # Runing time 2 days
#SBATCH --gpus=1                    # A number of GPUs  


singularity exec --bind /ist:/ist --bind /pure:/pure  --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ /ist/ist-share/vision/pakkapon/singularity/diffusionlight0230.sif python command_multi_illumn_real/runner.py --idx 4 --total 7