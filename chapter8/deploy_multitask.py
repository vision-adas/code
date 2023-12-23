import torch.onnx
import torch
from os.path import join
from chapter7.multitask_model_compress import MultiTaskCompress
print(torch.__version__)

# 指定设备，若目标设备为GPU，则指定为cuda:0，否则指定为cpu
device = "cpu"

# 指定模型文件夹
model_folder = "/home/vision/models/2023_10_01__17_49_41"

# 实例化模型
model = MultiTaskCompress(with_seg=True,
                          with_det=True,
                          with_uncertainty=True,
                          num_seg_classes=19,
                          backbone="resnet")
# 加载训练好的模型
model.load_weights(join(model_folder, "50.pth"))
# 指定阈值生成模型压缩方案，阈值越大，压缩率越高
compress_config = model.create_compression_configs(threshold=10e-7)
# 根据模型压缩方案压缩模型
model.compress(compress_config)
model.to(device)
model = model.eval()
torch.no_grad()
input = torch.rand((1, 480, 640, 3)).to(device)
# 对压缩后的模型进行推理测试
output = model(input)

"""导出torchscript模型"""
model_script = torch.jit.trace(model, input, strict=False)
# 冻结模型
model_script = torch.jit.freeze(model_script)
# 保存模型
torch.jit.save(model_script, join(model_folder, f"compressed_{device}.pt"))

"""导出onnx模型"""
# 加载torchscript模型，加载保存的模型更干净
model_to_export = torch.load(join(model_folder, f"compressed_{device}.pt"))
with torch.no_grad():
    model_to_export = model_to_export.eval()
    # 导出onnx模型
    torch.onnx.export(model_to_export,            # torchscript模型
                      input,                      # 用于推理的输入
                      join(model_folder, "compressed.onnx"),  # 模型保存路径
                      export_params=True,        # 是否导出模型参数
                      opset_version=11,          # 导出的onnx模型版本
                      do_constant_folding=True,  # 是否对模型进行常量折叠
                      input_names=['input'],     # 模型的输入名
                      output_names=["output"],   # 模型的输出名
                      dynamic_axes={'input': {0: 'batch_size'},    # 模型输入的可变维度
                                    'output': {0: 'batch_size'}})  # 模型输出的可变维度
