# 1. square to square hdr
# python 01_run_square2hdr.py

# 2. ball2envmap
cd /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/ball2perspectiveenvmap
../../bin/v100shell
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed100 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed100 --thread 25 -t 2 -i 0
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed100 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed100 --thread 25 -t 2 -i 1

python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed200 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed200 --thread 25 -t 2 -i 0
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed200 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed200 --thread 25 -t 2 -i 1

python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed300 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed300 --thread 25 -t 2 -i 0
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed300 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed300 --thread 25 -t 2 -i 1

python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed400 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed400 --thread 25 -t 2 -i 0
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed400 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed400 --thread 25 -t 2 -i 1

python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed500 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed500 --thread 25 -t 2 -i 0
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed500 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed500 --thread 25 -t 2 -i 1

python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed100 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed100 --thread 25 -t 8 -i 3
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed100 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed100 --thread 25 -t 8 -i 5

python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed200 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed200 --thread 25 -t 8 -i 3
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed200 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed200 --thread 25 -t 8 -i 5

python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed300 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed300 --thread 25 -t 8 -i 3
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed300 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed300 --thread 25 -t 8 -i 5

python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed400 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed400 --thread 25 -t 8 -i 3

################################
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed400 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed400 --thread 25 -t 8 -i 5

python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed500 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed500 --thread 25 -t 8 -i 3
python run_reverse_general.py --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed500 --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/seed500 --thread 25 -t 8 -i 5



# 3. convert from envmap to SH
cd ../envmap2shcoeff
python envmap2sh_hdr_general.py --threads 32 --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/  --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/ --scene_ids /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/run_multi_illumn_seed_small_portion/scenes.json --total 5 --idx 0

# 4. Rendering shading 
python efficient_sh_general.py --threads 32 --input_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/  --output_dir /pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination/ --scene_ids /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/run_multi_illumn_seed_small_portion/scenes.json --total 5 --idx 0

