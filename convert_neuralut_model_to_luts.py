import torch
import numpy as np

from neuralut.nn import (
    generate_truth_tables,
    module_list_to_verilog_module,
)

from neuralut_mnist_model import MnistNeqModel

add_registers = False

model_config = {
        "hidden_layers": [12, 12, 12],
        "input_length": 2,
        "output_length": 1,
        "input_bitwidth": 12,
        "hidden_bitwidth": 12,
        "output_bitwidth": 12,
        "input_fanin": 2,
        "hidden_fanin": 6,
        "output_fanin": 6,
        "width_n": 16,
        "batch_size": 12,
        "cuda": False
        }


model = MnistNeqModel(model_config)
if model_config["cuda"]:
    model.cuda()

checkpoint = torch.load('./log/h_next.pth', map_location="cuda:{}".format(model_config["device"]) if model_config["cuda"] else "cpu")
model.load_state_dict(checkpoint)#["model_dict"])

# Generate the truth tables in the LUT module
print("Converting to NEQs to LUTs...")
generate_truth_tables(model, verbose=True)

module_list_to_verilog_module(
    model.module_list,
    "neuralut",
    './log/',
    add_registers=add_registers,
)

