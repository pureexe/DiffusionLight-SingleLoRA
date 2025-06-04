#!/bin/sh

#SBATCH --error=output/jobs/err/p033.%j  
#SBATCH --output=output/jobs/out/p033.%j
#SBATCH --job-name=p033       # Job name
#SBATCH --mem=33GB                  # Memory request for this job
#SBATCH --nodes=1                   # The number of nodes
#SBATCH --partition=cpu
#SBATCH --account=vision
#SBATCH --time=72:0:0                # Runing time 2 days
#SBATCH --cpus-per-task 25

SINGULARITYENV_HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ \
SINGULARITYENV_HUB_HOME=/ist/ist-share/vision/huggingface/ \
SINGULARITYENV_HF_HUB_OFFLINE=1 \
singularity exec --bind /ist:/ist \
                 --bind /pure:/pure \
                 --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets \
                 --nv /pure/f1/singularity/relight_20250510.sif \
                 python run_reverse_general.py \
                 --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination_gt/train \
                 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination_gt/train \
                 --thread 25 -t 64 -i 33

