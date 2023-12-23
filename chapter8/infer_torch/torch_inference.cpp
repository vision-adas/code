#include "SegMetric.hpp"
#include <torch/torch.h>
#include <torch/script.h>
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
    // 加载语义分割数据集
    std::string image_folder = "../../data/camera/";
    std::string seg_folder = "../../data/seg/";
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
    // 加载模型
    torch::jit::Module module = torch::jit::load("compressed.pt");
    module.eval();
    torch::NoGradGuard no_grad;
    std::cout << "Model loaded, runing evaluation...\n";
    // 创建CPU设备
    torch::Device cpuDevice = torch::Device(torch::kCPU);
    // 创建CPU Tensor的选项对象
    auto options_cpu_int8 = torch::TensorOptions().dtype(torch::kUInt8).device(cpuDevice).requires_grad(false);

    // 读取数据集
    auto files = getSegGt();
    // 创建语义分割评价器
    SegMetric segMetric;
    float time = 0;
    for (int i = 0; i < files.size(); i++)
    {
        // 读取图像
        cv::Mat image = cv::imread(files[i].first);
        // 将OpenCV的BGR图像转换为RGB图像
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        std::vector<int64_t> shape{1, 480, 640, 3};
        // 基于图像数据的指针创建CPU Tensor
        at::Tensor tensor = torch::from_blob((void *)image.data, at::IntArrayRef(shape), options_cpu_int8);

        timer.start();
        // 前向推理
        c10::IValue output = module.forward({tensor});
        timer.stop();
        if (i > 5)
            time += timer.elapsed_time_ms();

        // 将输出Tensor进行内存对齐并转换为CPU Tensor
        at::Tensor outputTensor = output.toTensor().contiguous();
        at::Tensor seg = outputTensor.to(cpuDevice).toType(torch::kUInt8);

        // 基于Tensor的指针创建OpenCV Mat
        cv::Mat pred(120, 160, CV_8UC1, seg.data_ptr());

        // 保存预测结果
        auto save_seg = "seg_pred/" + zfill(i, 6) + ".png";
        cv::imwrite(save_seg, pred * 10u);
        std::cout << save_seg << std::endl;

        // 读取语义分割标签
        cv::Mat gt = cv::imread(files[i].second, 0);
        cv::resize(gt, gt, {160, 120}, 0, 0, cv::INTER_NEAREST);
        // 评价语义分割结果
        segMetric.append(pred, gt, 19);
    }

    std::cout << "mIoU:" << segMetric.compute_mIoU({0, 1, 2, 8, 10, 11, 12, 13}) << "\n";
    std::cout << "Time:" << time / ((float)files.size() - 6) << " ms\n";
}
