
TOTAL_JOB = 16

for i in range(TOTAL_JOB):
    
    output = f"""#!/bin/sh

#SBATCH --error=output/jobs/err/{i:03d}.%j  
#SBATCH --output=output/jobs/out/{i:03d}.%j
#SBATCH --job-name={i:03d}       # Job name
#SBATCH --mem=32GB                  # Memory request for this job
#SBATCH --nodes=1                   # The number of nodes
#SBATCH --partition=gpu-4080
#SBATCH --account=vision
#SBATCH --time=72:0:0                # Runing time 2 days
#SBATCH --gpus=1                    # A number of GPUs  

singularity exec --bind /ist:/ist  --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ /ist/ist-share/vision/pakkapon/singularity/diffusionlight0230.sif python command/runner.py --idx {i} --total {TOTAL_JOB}
"""
    with open(f"{i:03d}.sh",'w') as f:
        f.write(output)
        