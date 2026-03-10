import safetensors
import llaisys
from test_utils import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=None, type=str)
args = parser.parse_args()

model = llaisys.Qwen2(model_path=args.model_path)
print(model.model_path)
print(model.config)

for file in sorted(model.model_path.glob("*.safetensors")):
    data_ = safetensors.safe_open(file, framework="pt", device="cpu")
    for name_ in data_.keys():
        tensor_ = data_.get_tensor(name_)

        assert check_equal(model.weights[name_], tensor_)



