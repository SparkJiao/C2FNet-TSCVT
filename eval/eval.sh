CUDA_VISIBLE_DEVICES=0 python main.py --methods s2+s0 --save_dir ./results --root_dir ~/PycharmProjects/TSCVT_code/eval --datasets CAMO+CHAMELEON+COD10K

CUDA_VISIBLE_DEVICES=0 python main.py --methods C2FNet49 --output_dir ../results/ --data_dir ../data/TestDataset --datasets CAMO+CHAMELEON+COD10K
