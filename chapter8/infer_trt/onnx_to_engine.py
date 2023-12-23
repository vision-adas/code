"""
onnx graph to tensorRT engine conversion script

usage: CUDA_VISIBLE_DEVICES=0 python onnx_to_trt.py --onnx some_seg.onnx --save_path some_seg.engine --precision 8
"""
from typing import Tuple, Dict

import argparse
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
print(trt.__version__)
# set trt logger to verbose
TRT_LOGGER = trt.Logger()
TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE


def parse_args():
    """Parse arguments for onnx to trt conversion"""
    parser = argparse.ArgumentParser(description='onnx to tensorRT converter')
    parser.add_argument('--onnx_file', type=str,
                        default="/workspace/compressed.onnx", help='input onnx file path')
    parser.add_argument('--save_path', type=str,
                        default="/workspace/compressed_fp32.engine", help='trt engine save path+name')
    parser.add_argument('--precision', type=int, default=8,
                        help='trt engine precision, 8, 16 and 32', choices=[8, 16, 32])
    parser.add_argument('--num_samples', type=int, default=100,
                        help='num samples for calibration, only needed for precision 8')
    args = parser.parse_args()
    return vars(args)


def parse_onnx(onnx_file: str, parser: trt.OnnxParser):
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print('ERROR: Failed to parse the ONNX file: {}'.format(onnx_file))
            for error in range(parser.num_errors):
                print(parser.get_error(error))


def mark_outputs(network):
    # Mark last layer's outputs if not already marked
    # NOTE: This may not be correct in all cases
    last_layer = network.get_layer(network.num_layers-1)
    if not last_layer.num_outputs:
        print("Last layer contains no outputs.")
        return

    for i in range(last_layer.num_outputs):
        network.mark_output(last_layer.get_output(i))


def check_network(network):
    assert network.num_layers, "no layers found, invalid network"

    if not network.num_outputs:
        print("No output nodes found, marking last layer's outputs as network outputs")
        mark_outputs(network)

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] +
                  [len(out.name) for out in outputs])

    print("=== Network Description ===")
    for i, inp in enumerate(inputs):
        print("Input  {0} | Name: {1:{2}} | Shape: {3}".format(
            i, inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        print("Output {0} | Name: {1:{2}} | Shape: {3}".format(
            i, out.name, max_len, out.shape))


class RndInt8Calibrator(trt.IInt8EntropyCalibrator):
    """trtexec 默认"""

    def __init__(self, inputs: Dict[str, Tuple[int]],
                 num_samples: int):
        """初始化函数

        Args:
            inputs (Dict[str, Tuple[int]]): 输入张量的名字和维度
            num_samples (int): 用于校准的样本数
        """
        trt.IInt8EntropyCalibrator.__init__(self)
        self.inputs = inputs
        self.num_samples = num_samples
        self.current_sample = 0
        self.batch_size = 1

        self.volumes = []
        self.host_inputs = []
        self.device_inputs = []
        for input_name, input_shape in self.inputs.items():
            # 检查张量是否为空
            count = len(list(filter(lambda x: x < 1, input_shape[1:])))
            assert count == 0, "input shapes are not defined, got {}".format(
                input_shape)

            # 计算空间占用
            volume = trt.volume((self.batch_size, *input_shape[1:]))
            self.volumes.append(volume)
            # 为输入分配内存空间
            self.host_inputs.append(cuda.pagelocked_empty(volume, np.float32))
            # 为输入分配显存空间
            self.device_inputs.append(
                cuda.mem_alloc(volume * trt.float32.itemsize))

    def get_batch_size(self):
        # 此为必须实现的接口函数，返回batch size
        return self.batch_size

    def get_batch(self, names):
        # 此为必须实现的接口函数，返回一个batch的输入数据
        if self.current_sample >= self.num_samples:
            return None
        self.current_sample += self.batch_size
        # 为所有的输入张量生成数据
        for idx, volume in enumerate(self.volumes):
            # 生成一个取值范围为0到1的随机输入张量
            data = np.random.uniform(
                low=0, high=255, size=volume).astype(np.float32)
            # 将Numpy数组的数据整理为连续内存并复制到内存
            np.copyto(self.host_inputs[idx], data.ravel())
            # 将内存数据复制到显存
            cuda.memcpy_htod(self.device_inputs[idx], self.host_inputs[idx])

        # 返回输入数据的显存空间
        return [int(device_input) for device_input in self.device_inputs]

    def read_calibration_cache(self):
        pass

    def write_calibration_cache(self, cache):
        pass


def onnx_to_trt(onnx_file: str, save_path: str, precision: int, num_samples: int):
    # TRT variables
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(network_flags)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 480, 640, 3),
                      (1, 480, 640, 3), (1, 480, 640, 3))
    config.add_optimization_profile(profile)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    runtime = trt.Runtime(TRT_LOGGER)

    # parse onnx graph
    parse_onnx(onnx_file, parser)
    check_network(network)

    config.max_workspace_size = 10 * 1 << 30   # 5 GB
    if precision == 8:
        assert builder.platform_has_fast_int8, "INT8 is not supported on this platform"
        config.set_flag(trt.BuilderFlag.INT8)
        inputs = {network.get_input(idx).name: network.get_input(idx).shape
                  for idx in range(network.num_inputs)}
        config.int8_calibrator = RndInt8Calibrator(
            inputs=inputs, num_samples=num_samples)
    if precision == 16:
        assert builder.platform_has_fast_fp16, "FP16 is not supported on this platform"
        config.set_flag(trt.BuilderFlag.FP16)

    # create and serialize engine
    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)
    with open(save_path, "wb") as out_file:
        print("Serializing engine to file: {:}".format(save_path))
        out_file.write(engine.serialize())


if __name__ == '__main__':
    args = parse_args()
    onnx_to_trt(**args)
