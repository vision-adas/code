#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "HDTimer.h"
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

std::string zfill(int num, int zeros)
{
    std::ostringstream str;
    str << std::setw(zeros) << std::setfill('0') << num;
    return str.str();
}

std::vector<std::pair<std::string, std::string>> getSegGt()
{
    std::string image_folder = "/data/camera/";
    std::string seg_folder = "/data/seg/";
    std::vector<std::pair<std::string, std::string>> gt_file_pairs = {};
    for (int i = 6400; i < 7600; i += 3)
    {
        std::string base = zfill(i, 6);
        gt_file_pairs.push_back({image_folder + base + ".jpg", seg_folder + base + ".png"});
    }
    return gt_file_pairs;
}

int main()
{
    HDTimer timer;
    if (!(torch::cuda::is_available() && torch::cuda::cudnn_is_available()))
    {
        std::cerr << "Cuda not avaliable, check if cuda libraries are avaliable\n";
        return false;
    }

    // 创建GPU设备
    torch::Device cudaDevice = torch::Device(torch::kCUDA, 0);
    // 加载模型
    torch::jit::Module module = torch::jit::load("compressed_cuda.pt");
    module.to(cudaDevice);
    module.to(at::kCUDA);
    module.eval();
    torch::NoGradGuard no_grad;
    // 创建CPU设备
    torch::Device cpuDevice = torch::Device(torch::kCPU);
    // 创建CPU Tensor的选项对象
    auto options_cpu_int8 = torch::TensorOptions().dtype(torch::kUInt8).device(cpuDevice).requires_grad(false);
    // 读取数据集
    auto gt_files = getSegGt();
    float time = 0;
    for (int index = 0; index < gt_files.size(); index++)
    {
        // 读取图像
        cv::Mat image = cv::imread(gt_files[index].first);
        // 将OpenCV的BGR图像转换为RGB图像
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        std::vector<int64_t> shape{1, 480, 640, 3};
        // 基于图像数据的指针创建CPU Tensor
        at::Tensor tensor = torch::from_blob((void *)image.data, at::IntArrayRef(shape), options_cpu_int8);
        // 将CPU Tensor转换为GPU Tensor
        tensor = tensor.to(cudaDevice);
        timer.start();
        // 前向推理
        c10::IValue output = module.forward({tensor});
        timer.stop();
        if (index > 5)
            time += timer.elapsed_time_ms();
        // 将Tensor进行内存对齐并转换为CPU Tensor
        at::Tensor outputTensor = output.toTensor().contiguous();
        at::Tensor seg = outputTensor.to(cpuDevice).toType(torch::kUInt8);

        // 基于Tensor的指针创建OpenCV Mat
        cv::Mat pred(120, 160, CV_8UC1, seg.data_ptr());

        // 保存预测结果
        auto save_seg = "seg_pred/" + zfill(index, 6) + ".png";
        cv::imwrite(save_seg, 10 * pred);
        std::cout << save_seg << "\n";
    }
}