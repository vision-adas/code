# 《PyTorch自动驾驶视觉感知算法实战》示例代码
本代码库中的代码将在本书上市后继续更新和改进，包括适配更新的版本，修改bug，对书中的错漏之处进行勘误，甚至根据读者要求添加新功能。代码库中的代码可以直接训练和调试，代码亦可以用于商业用途。

gitee代码库：https://gitee.com/vision-adas/code

代码库的文件内容如下：

- chapter2：PyTorch示例代码，包括一个用于示例的Jupyter Notebook和一个基本的图像分类示例
- chapter4: Yolo目标检测和TorchVision目标检测库的训练代码
- chapter5: 语义分割的训练代码
- chapter6: 无监督单目深度估计的训练代码
- chapter7: 目标检测和语义分割多任务网络的训练代码
- chapter8: 模型导出脚本以及LibTorch、TensorRT的模型部署C++代码
- configs: 一些项目配置文件
- detection_torchvision: 用于TorchVision目标检测模型训练的复制代码，来自TorchVision官方代码库的Reference文件。
- lib: 一些公用的类
- requirements.txt: 用于配置CPU版PyTorch的配置文件
- requirements_gpu.txt: 用于配置GPU版PyTorch的配置文件

## 环境配置
### 配置Python开发环境

配置的环境为
Python3.8+Pytorch2.0.0+TorchVision0.15.0+CPU

1. 下载安装Anaconda后，执行以下代码新建一个虚拟环境：

```bash
conda create -n book python=3.8
```

2. 环境建立成功后激活环境:

```bash
conda acrtivate book
```

3. 安装依赖项

```bash
pip install -r requirements.txt
```
4. 检查是否成功
```bash
cd chapter2
python train.py
```
### 下载数据集

Onedrive下载链接：https://1drv.ms/u/s!AiFavgbAhwpFb4H9uUxemMHQlbg

下载后得到一个munich_dataset.zip的1.7GB大小文件，将文件解压至data文件夹：

```bash
unzip munich_dataset.zip -d data
```
文件夹结构为

- data

  - camera
  - detections
  - seg
  - seg_simple

### 训练与部署

#### 第四章：训练目标检测模型

在代码库的根目录下，执行以下命令即可开始训练Yolo：
```bash
PYTHONPATH=. python chapter4/train_yolo.py
```
执行以下命令即可开始训练TorchVision提供的FasterRCNN：
```bash
PYTHONPATH=. python chapter4/det_demo_torchvision.py
```

#### 第五章：训练语义分割模型

在代码库的根目录下，执行以下命令即可开始训练语义分割模型：
```bash
PYTHONPATH=. python chapter5/train_seg.py
```

#### 第六章: 训练无监督单目深度模型

在代码库的根目录下，执行以下命令即可开始无监督单目深度模型的训练：
```bash
PYTHONPATH=. python chapter6/train_depth.py
```

#### 第七章：训练多任务网络模型

在代码库的根目录下，执行以下命令即可开始训练Yolo和语义分割多任务模型：
```bash
PYTHONPATH=. python chapter7/train_multitask.py
```

#### 第八章：进行模型部署

部署模型之前，请先完成第七章多任务网络模型的训练，得到训练好的pth模型文件。

##### 将模型导出为.pt和.onnx格式

打开chapter8/deploy_multitask.py文件，进行以下设置：

- 将第七章的训练文件夹路径设置为model_folder
- 默认使用最后一个epoch的模型，也就是50.pth，若epoch数不为50，则将50.pth改为最后一个epoch的模型文件名。
- 设置训练模型的设备，"cpu"或"cuda:0"
- 设置输出文件的保存路径

执行以下命令进行模型导出：
```bash
PYTHONPATH=. python chapter8/deploy_multitask.py
```

##### 使用LibTorch部署模型

从 [download.pytorch.org/libtorch/cpu/](https://download.pytorch.org/libtorch/cpu/) 页面下载LibTorch 2.0.1 的CPU版本，解压至 chapter8/infer_torch 文件夹

按照顺序执行以下命令编译执行LibTorch模型

```bash
cd chapter8
docker build -t vision .
cd ..
docker run -v $(pwd):/workspace -w /workspace -it vision bash
cd chapter8/infer_torch/
mkdir build
cd build
make
cd ..
mkdir seg_pred
./build/torch_inference
```

执行完成后seg_pred文件夹中应出现推理输出的语义分割图。

##### 使用TensorRT部署模型

由于本书示例的Docker镜像是TensorRT的镜像，因此无需安装TensorRT，可以直接按顺序执行以下命令：

```bash
docker run --gpus all -v $(pwd):/workspace -w /workspace -it vision bash
cd chapter8/infer_trt/
mkdir build
cd build
make
cd ..
mkdir seg_pred
./build/trt_inference
```
