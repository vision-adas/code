#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <numeric>
#include <algorithm>
#include <NvInfer.h>
#include <NvCaffeParser.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include "SegMetric.hpp"
#include "HDTimer.h"

std::string zfill(int num, int zeros)
{
    std::ostringstream str;
    str << std::setw(zeros) << std::setfill('0') << num;
    return str.str();
}

std::vector<std::pair<std::string, std::string>> getSegGt()
{
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

const std::string TrtValueTypes[5] = {"float", "half", "int8", "int32", "bool"};

inline int64_t trtTensorVolume(const nvinfer1::Dims &d, const int batchSize, const nvinfer1::DataType &dType)
{
    // 计算Tensor的总大小
    size_t bytes;
    if (dType == nvinfer1::DataType::kFLOAT or dType == nvinfer1::DataType::kINT32)
        bytes = 4;
    else if (dType == nvinfer1::DataType::kHALF)
        bytes = 2;
    else if (dType == nvinfer1::DataType::kINT8)
        bytes = 1;
    else
        throw std::runtime_error("Invalid tensorRT DataType.");

    return bytes * batchSize * std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

class TrtLogger : public nvinfer1::ILogger
{
    // 根据TensorRT的日志接口，实现自己的日志类
    void log(Severity severity, const char *msg) noexcept override
    {
        try
        {
            // only print info-level messages
            if (severity != Severity::kVERBOSE)
                std::cout << msg << std::endl;
        }
        catch (std::ios_base::failure &e) // normally not used, but could be enabled
        {
            // ignore exceptions
        }
    }
};

int main()
{
    TrtLogger logger;
    nvinfer1::ICudaEngine *mEngine = nullptr;
    nvinfer1::IExecutionContext *mContext = nullptr;

    // 从文件中加载TensorRT引擎
    std::string engineFile = "compressed_fp32.engine";

    std::cout << "Load engine file " << engineFile << "\n";
    std::ifstream in(engineFile.c_str(), std::ifstream::binary);
    if (in.is_open())
    {
        auto const start_pos = in.tellg();
        in.ignore(std::numeric_limits<std::streamsize>::max());
        size_t bufCount = in.gcount();
        in.seekg(start_pos);
        std::unique_ptr<char[]> engineBuf(new char[bufCount]);
        in.read(engineBuf.get(), bufCount);
        // 初始化TensorRT插件，我们不使用插件，所以不执行这一步
        // initLibNvInferPlugins(&mLogger, "");
        // 创建TensorRT运行时
        auto iRuntime = nvinfer1::createInferRuntime(logger);
        std::cout << "Deserialize cuda engine\n";
        // 反序列化TensorRT引擎
        mEngine = iRuntime->deserializeCudaEngine((void *)engineBuf.get(), bufCount, nullptr);

#ifdef kNV_TENSORRT_VERSION_IMPL // --- version > 8
        delete iRuntime;
#else
        iRuntime->destroy();
#endif
    }
    // 创建TensorRT上下文
    mContext = mEngine->createExecutionContext();
    // 获取TensorRT引擎的输入输出Binding
    int nbBindings = mEngine->getNbBindings();

    std::vector<void *> mBinding;
    std::vector<size_t> mBindingSize;

    mBinding.resize(nbBindings);
    mBindingSize.resize(nbBindings);
    if (!mContext->allInputShapesSpecified())
    { // 检查是否有动态输入
        std::cerr << "dynamic input shapes found, they must be set using set_shape_input()\n";
        return false;
    }
    // 获取Batch大小
    int32_t mBatchSize = mEngine->getBindingDimensions(0).d[0];

    for (int i = 0; i < nbBindings; i++)
    {
        // 获取TensorRT引擎Binding的维度、数据类型、名称
        nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        std::string nodeName = std::string(mEngine->getBindingName(i));
        // 获取Binding所需要的存储空间大小
        int64_t totalSize = trtTensorVolume(dims, mBatchSize, dtype);
        mBindingSize[i] = totalSize;
        std::cout << nodeName << ":" << dims.d[0] << "," << dims.d[1] << "," << dims.d[2] << "," << dims.d[3] << "\n";
        std::cout << totalSize << "\n";
        // 为Binding分配存储空间
        cudaMalloc((void **)&mBinding[i], totalSize);
    }

    auto files = getSegGt();
    auto segMetric = SegMetric();
    cv::Mat predSeg = cv::Mat::zeros(120, 160, CV_32SC1);
    cv::Mat predSegUint8 = cv::Mat::zeros(120, 160, CV_8UC1);
    HDTimer timer;
    float time = 0;
    for (int i = 0; i < files.size(); i++)
    {
        cv::Mat image = cv::imread(files[i].first);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        cv::Mat floatImage;
        image.convertTo(floatImage, CV_32FC3);
        // 将图像数据拷贝到Binding对应的GPU空间
        cudaMemcpy(mBinding[0], (void *)floatImage.data, mBindingSize[0], cudaMemcpyHostToDevice);

        timer.start();
        mContext->execute(1, &mBinding[0]);
        timer.stop();

        // 将输出拷贝到CPU空间
        cudaMemcpy((void *)predSeg.data, mBinding[1], mBindingSize[1], cudaMemcpyDeviceToHost);
        std::cout << timer.elapsed_time_ms() << "\n";
        if (i > 5)
            time += timer.elapsed_time_ms();
        predSeg.convertTo(predSegUint8, CV_8UC1);

        cv::Mat gt = cv::imread(files[i].second, 0);
        cv::resize(gt, gt, {160, 120}, 0, 0, cv::INTER_NEAREST);
        segMetric.append(predSegUint8, gt, 19);
    }
    std::cout << "-------------------\n";
    std::cout << time / (float)(files.size() - 6) << "\n";
    std::cout << segMetric.compute_mIoU({0, 1, 2, 8, 10, 11, 12, 13}) << "\n";
    return 0;
}
