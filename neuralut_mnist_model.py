#  This file is part of NeuraLUT.
#  
#  NeuraLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.

#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import reduce
from os.path import realpath

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init

from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.nn import QuantHardTanh, QuantReLU

from neuralut.quant import QuantBrevitasActivation
from neuralut.nn import (
    SparseLinearNeq,
    ScalarBiasScale,
    FeatureMask,
)
from neuralut.init import random_restrict_fanin


class MnistNeqModel(nn.Module):
    def __init__(self, model_config):
        super(MnistNeqModel, self).__init__()
        self.model_config = model_config
        self.is_cuda = model_config["cuda"]
        self.num_neurons = (
            [model_config["input_length"]]
            + model_config["hidden_layers"]
            + [model_config["output_length"]]
        )
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i - 1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                # input_bias = ScalarBiasScale(scale=False, bias_init=-0.25)
                input_quant = QuantBrevitasActivation(
                    QuantHardTanh(
                        bit_width=model_config["input_bitwidth"],
                        min_val=0.0,
                        max_val=1.0,
                        narrow_range=False,
                        quant_type=QuantType.INT,
                        scaling_impl_type=ScalingImplType.PARAMETER,
                    ),
                    pre_transforms=[bn_in]#, input_bias],
                )
                output_quant = QuantBrevitasActivation(
                    QuantReLU(
                        bit_width=model_config["hidden_bitwidth"],
                        max_val=1.61,
                        quant_type=QuantType.INT,
                        scaling_impl_type=ScalingImplType.PARAMETER,
                    ),
                    pre_transforms=[bn],
                )
                imask = FeatureMask(
                    in_features,
                    out_features,
                    fan_in=model_config["input_fanin"],
                    cuda=model_config["cuda"],
                )
                layer = SparseLinearNeq(
                    in_features,
                    out_features,
                    input_quant=input_quant,
                    output_quant=output_quant,
                    imask=imask,
                    fan_in=model_config["input_fanin"],
                    width_n=model_config["width_n"],
                    cuda=model_config["cuda"],
                )
                layer_list.append(layer)
            elif i == len(self.num_neurons) - 1:
                # output_bias_scale = ScalarBiasScale(bias_init=0.33)
                output_quant = QuantBrevitasActivation(
                    QuantHardTanh(
                        bit_width=model_config["output_bitwidth"],
                        min_val=0.0,
                        max_val=1.0,
                        narrow_range=False,
                        quant_type=QuantType.INT,
                        scaling_impl_type=ScalingImplType.PARAMETER,
                    ),
                    pre_transforms=[bn],
                    # post_transforms=[output_bias_scale],
                )
                imask = FeatureMask(
                    in_features,
                    out_features,
                    fan_in=model_config["output_fanin"],
                    cuda=model_config["cuda"],
                )
                layer = SparseLinearNeq(
                    in_features,
                    out_features,
                    input_quant=layer_list[-1].output_quant,
                    output_quant=output_quant,
                    imask=imask,
                    fan_in=model_config["output_fanin"],
                    width_n=model_config["width_n"],
                    apply_input_quant=False,
                    cuda=model_config["cuda"],
                )
                layer_list.append(layer)
            else:
                output_quant = QuantBrevitasActivation(
                    QuantReLU(
                        bit_width=model_config["hidden_bitwidth"],
                        max_val=1.61,
                        quant_type=QuantType.INT,
                        scaling_impl_type=ScalingImplType.PARAMETER,
                    ),
                    pre_transforms=[bn],
                )
                imask = FeatureMask(
                    in_features,
                    out_features,
                    fan_in=model_config["hidden_fanin"],
                    cuda=model_config["cuda"],
                )
                layer = SparseLinearNeq(
                    in_features,
                    out_features,
                    input_quant=layer_list[-1].output_quant,
                    output_quant=output_quant,
                    imask=imask,
                    fan_in=model_config["hidden_fanin"],
                    width_n=model_config["width_n"],
                    apply_input_quant=False,
                    cuda=model_config["cuda"],
                )
                layer_list.append(layer)
        self.module_list = nn.ModuleList(layer_list)
        self.is_verilog_inference = False
        self.latency = 1
        self.verilog_dir = None
        self.top_module_filename = None
        self.dut = None
        self.logfile = None

    def pytorch_inference(self):
        self.is_verilog_inference = False

    def verilog_forward(self, x):
        # Get integer output from the first layer
        input_quant = self.module_list[0].input_quant
        output_quant = self.module_list[-1].output_quant
        _, input_bitwidth = self.module_list[0].input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.module_list[-1].output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.module_list[0].in_features * input_bitwidth
        total_output_bits = self.module_list[-1].out_features * output_bitwidth
        num_layers = len(self.module_list)
        input_quant.bin_output()
        self.module_list[0].apply_input_quant = False
        y = torch.zeros(x.shape[0], self.module_list[-1].out_features)
        x = input_quant(x)
        self.dut.io.rst = 0
        self.dut.io.clk = 0
        for i in range(x.shape[0]):
            x_i = x[i, :]
            y_i = self.pytorch_forward(x[i : i + 1, :])[0]
            xv_i = list(map(lambda z: input_quant.get_bin_str_from_int(z, self.is_cuda), x_i))
            ys_i = list(map(lambda z: output_quant.get_bin_str_from_int(z, self.is_cuda), y_i))
            xvc_i = reduce(lambda a, b: a + b, xv_i[::-1])
            ysc_i = reduce(lambda a, b: a + b, ys_i[::-1])
            self.dut["M0"] = int(xvc_i, 2)
            for j in range(self.latency + 1):
                res = self.dut[f"M{num_layers}"]
                result = f"{res:0{int(total_output_bits)}b}"
                self.dut.io.clk = 1
                self.dut.io.clk = 0
            expected = f"{int(ysc_i,2):0{int(total_output_bits)}b}"
            result = f"{res:0{int(total_output_bits)}b}"
            assert expected == result
            res_split = [
                result[i : i + output_bitwidth]
                for i in range(0, len(result), output_bitwidth)
            ][::-1]
            yv_i = torch.Tensor(list(map(lambda z: int(z, 2), res_split)))
            y[i, :] = yv_i
            # Dump the I/O pairs
            if self.logfile is not None:
                with open(self.logfile, "a") as f:
                    f.write(
                        f"{int(xvc_i,2):0{int(total_input_bits)}b}{int(ysc_i,2):0{int(total_output_bits)}b}\n"
                    )
        return y

    def pytorch_forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x

    def forward(self, x):
        if self.is_verilog_inference:
            return self.verilog_forward(x)
        else:
            return self.pytorch_forward(x)


class MnistLutModel(MnistNeqModel):
    pass


class MnistVerilogModel(MnistNeqModel):
    pass
