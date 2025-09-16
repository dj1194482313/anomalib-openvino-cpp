#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "utils.h"

using namespace std;
using namespace cv;

/**
 * @brief PatchCore异常检测器类
 * 
 * 封装OpenVINO推理逻辑，支持图像预处理、异常图后处理和分数计算
 * 提供完整的检测结果结构体和可视化功能
 */
class PatchCoreDetector {
private:
    bool openvino_preprocess_;       // 是否使用OpenVINO图片预处理
    bool efficient_ad_;              // 是否使用efficient_ad模型
    string model_path_;              // 模型文件路径
    string metadata_path_;           // 元数据文件路径
    string device_;                  // 推理设备 (CPU/GPU)
    
    MetaData metadata_;              // 模型元数据
    ov::CompiledModel compiled_model_;       // 编译好的模型
    ov::InferRequest infer_request_;         // 推理请求
    vector<ov::Output<const ov::Node>> inputs_;  // 模型输入列表
    vector<ov::Output<const ov::Node>> outputs_; // 模型输出列表

    /**
     * @brief 初始化OpenVINO模型
     * @param model_path 模型文件路径
     * @param device 推理设备
     */
    void initializeModel(const string& model_path, const string& device);
    
    /**
     * @brief 模型预热
     */
    void warmUp();
    
    /**
     * @brief 执行推理
     * @param image 输入图像
     * @return 推理结果
     */
    Result performInference(Mat& image);

public:
    /**
     * @brief 构造函数
     * @param model_path 模型文件路径
     * @param metadata_path 元数据文件路径  
     * @param device 推理设备 (CPU/GPU)
     * @param use_openvino_preprocess 是否使用OpenVINO预处理
     * @param is_efficient_ad 是否为efficient_ad模型
     */
    PatchCoreDetector(const string& model_path, 
                     const string& metadata_path,
                     const string& device = "CPU",
                     bool use_openvino_preprocess = true,
                     bool is_efficient_ad = false);
    
    /**
     * @brief 析构函数
     */
    ~PatchCoreDetector() = default;
    
    /**
     * @brief 单张图片异常检测
     * @param image 输入图像 (RGB格式)
     * @param threshold 异常阈值
     * @return 检测结果，包含异常图和分数
     */
    Result detectSingle(Mat& image, float threshold = 0.5);
    
    /**
     * @brief 批量图片异常检测
     * @param image_dir 图片文件夹路径
     * @param save_dir 结果保存路径
     * @param threshold 异常阈值
     * @return 平均推理时间(毫秒)
     */
    double detectBatch(const string& image_dir, const string& save_dir, float threshold = 0.5);
    
    /**
     * @brief 获取模型信息
     * @return 模型信息字符串
     */
    string getModelInfo() const;
    
    /**
     * @brief 设置异常检测阈值
     * @param image_threshold 图像级阈值
     * @param pixel_threshold 像素级阈值
     */
    void setThresholds(float image_threshold, float pixel_threshold);
    
    /**
     * @brief 获取支持的设备列表
     * @return 设备名称列表
     */
    static vector<string> getAvailableDevices();
};