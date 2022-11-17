import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
args = parser.parse_args()

state_dict = torch.load(args.input_file, map_location="cpu")

cnt = 0
for _, param in state_dict.items():
    cnt += param.numel()

print(cnt)

