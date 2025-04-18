#!/bin/sh

#SBATCH --error=output/jobs/err/h008.%j  
#SBATCH --output=output/jobs/out/h008.%j
#SBATCH --job-name=h008       # Job name
#SBATCH --mem=32GB                  # Memory request for this job
#SBATCH --nodes=1                   # The number of nodes
#SBATCH --partition=gpu-cluster
#SBATCH --account=vision
#SBATCH --time=72:0:0                # Runing time 2 days
#SBATCH --gpus=1                    # A number of GPUs  

singularity exec --bind /ist:/ist  --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ /ist/ist-share/vision/pakkapon/singularity/diffusers0310v6.sif python run_square2hdr.py -i 8 -t 40
