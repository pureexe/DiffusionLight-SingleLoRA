#!/bin/sh

#SBATCH --error=output/jobs/err/master.%j  
#SBATCH --output=output/jobs/out/master.%j
#SBATCH --job-name=master       # Job name
#SBATCH --mem=755GB                  # Memory request for this job
#SBATCH --nodes=1                   # The number of nodes
#SBATCH --partition=cpu
#SBATCH --account=vision
#SBATCH --cpus-per-task=80
#SBATCH --time=72:0:0                # Runing time 2 days

#singularity exec --bind /ist:/ist  --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ /ist/ist-share/vision/pakkapon/singularity/diffusers0310v6.sif python run_reverse.py --idx 0 --total 40

SINGULARITYENV_HF_HUB_CACHE='/ist/ist-share/vision/huggingface/hub/' \
SINGULARITYENV_HUB_HOME='/ist/ist-share/vision/huggingface/' \
SINGULARITYENV_HF_HOME="/ist/users/$USER/.cache/huggingface" \
SINGULARITYENV_PYTHONPATH="$(pwd):$(pwd)/src" \
singularity exec --bind /ist:/ist \
  --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets \
  /ist/ist-share/vision/pakkapon/singularity/diffusers0310v6.sif \ 
  python run_reverse.py --idx 0 --total 40
