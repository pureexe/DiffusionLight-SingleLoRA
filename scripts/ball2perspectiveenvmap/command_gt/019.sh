#!/bin/sh

#SBATCH --error=output/jobs/err/p019.%j  
#SBATCH --output=output/jobs/out/p019.%j
#SBATCH --job-name=p019       # Job name
#SBATCH --mem=32GB                  # Memory request for this job
#SBATCH --nodes=1                   # The number of nodes
#SBATCH --partition=gpu-cluster
#SBATCH --account=vision
#SBATCH --time=72:0:0                # Runing time 2 days
#SBATCH --gpus=1                    # A number of GPUs  
#SBATCH --gpus=1                    # A number of GPUs  
#SBATCH --cpus-per-task 25

singularity exec --bind /ist:/ist --bind /pure:/pure --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ --env HF_HUB_OFFLINE=1 /pure/f1/singularity/relight_20250510.sif python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination_gt/train --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination_gt/train --thread 25 -t 64 -i 19

