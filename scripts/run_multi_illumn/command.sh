# 1. square to square_hdr
bin/v100shell
python scripts/run_multi_illumn/run_square2hdr.py --total 1 --idx 0
python run_marigold.py --total 17 --idx 0

# if not have fov. need to run fov first
cd /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/
bin/v100
cd scripts/ball2perspectiveenvmap
python run_reverse_general.py  --threads 25 --total 16 --idx 0

# convert from envmap to SH
cd ../envmap2shcoeff
python envmap2sh_hdr_general.py --threads 32 --total 16 --idx 0

# create shading
cd ../efficient_rendering
python efficient_sh_general.py --total 4 --idx 0