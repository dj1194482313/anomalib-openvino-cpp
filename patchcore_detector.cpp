#include "patchcore_detector.h"
#include <iostream>
#include <filesystem>
#include <numeric>

PatchCoreDetector::PatchCoreDetector(const string& model_path, 
                                   const string& metadata_path,
                                   const string& device,
                                   bool use_openvino_preprocess,
                                   bool is_efficient_ad) 
    : model_path_(model_path)
    , metadata_path_(metadata_path)
    , device_(device)
    , openvino_preprocess_(use_openvino_preprocess)
    , efficient_ad_(is_efficient_ad) {
    
    cout << "[INFO] Initializing PatchCore Detector..." << endl;
    cout << "[INFO] Model: " << model_path_ << endl;
    cout << "[INFO] Metadata: " << metadata_path_ << endl;
    cout << "[INFO] Device: " << device_ << endl;
    
    // 1. 加载元数据
    metadata_ = getJson(metadata_path_);
    
    // 2. 初始化模型
    initializeModel(model_path_, device_);
    
    // 3. 获取输入输出信息
    inputs_ = compiled_model_.inputs();
    outputs_ = compiled_model_.outputs();
    
    // 4. 打印模型信息
    cout << getModelInfo() << endl;
    
    // 5. 创建推理请求
    infer_request_ = compiled_model_.create_infer_request();
    
    // 6. 模型预热
    warmUp();
    
    cout << "[INFO] PatchCore Detector initialized successfully!" << endl;
}

void PatchCoreDetector::initializeModel(const string& model_path, const string& device) {
    try {
        // 初始化OpenVINO Runtime core
        ov::Core core;
        
        // 从文件读取模型
        shared_ptr<ov::Model> model = core.read_model(model_path);
        
        if (openvino_preprocess_) {
            vector<float> mean, std;
            
            if (!efficient_ad_) {
                // ImageNet标准化参数
                mean = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
                std = {0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f};
            } else {
                // EfficientAD模型参数
                mean = {0.0f, 0.0f, 0.0f};
                std = {255.0f, 255.0f, 255.0f};
            }
            
            // 配置预处理管道
            ov::preprocess::PrePostProcessor ppp(model);
            
            // 设置输入图像格式
            ppp.input(0).tensor()
                .set_color_format(ov::preprocess::ColorFormat::RGB)
                .set_element_type(ov::element::u8)
                .set_layout(ov::Layout("HWC"));
            
            // 设置预处理步骤
            ppp.input(0).preprocess()
                .convert_element_type(ov::element::f32)
                .mean(mean)
                .scale(std);
            
            // 设置模型输入布局
            ppp.input(0).model().set_layout(ov::Layout("NCHW"));
            
            // 设置输出格式
            for (size_t i = 0; i < model->outputs().size(); i++) {
                ppp.output(i).tensor().set_element_type(ov::element::f32);
            }
            
            // 构建预处理管道
            model = ppp.build();
        }
        
        // 编译模型到设备
        compiled_model_ = core.compile_model(model, device);
        
    } catch (const exception& e) {
        throw runtime_error("[ERROR] Failed to initialize model: " + string(e.what()));
    }
}

void PatchCoreDetector::warmUp() {
    cout << "[INFO] Warming up model..." << endl;
    
    // 创建dummy输入数据
    Size size(metadata_.infer_size[1], metadata_.infer_size[0]);
    Mat dummy_input = Mat::zeros(size, CV_8UC3);
    
    // 执行预热推理
    performInference(dummy_input);
    
    cout << "[INFO] Model warm-up completed!" << endl;
}

Result PatchCoreDetector::performInference(Mat& image) {
    // 保存原始图像尺寸
    metadata_.image_size[0] = image.rows;
    metadata_.image_size[1] = image.cols;
    
    // 图像预处理
    Mat blob;
    if (openvino_preprocess_) {
        // 使用OpenVINO预处理，只需调整尺寸
        resize(image, blob, Size(metadata_.infer_size[1], metadata_.infer_size[0]));
    } else {
        // 使用OpenCV预处理
        blob = pre_process(image, metadata_, efficient_ad_);
        blob = dnn::blobFromImage(blob);
    }
    
    // 创建输入tensor
    size_t input_size = compiled_model_.input(0).get_byte_size();
    ov::Tensor input_tensor(compiled_model_.input(0).get_element_type(),
                           compiled_model_.input(0).get_shape(),
                           blob.data);
    
    // 执行推理
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.infer();
    
    // 获取异常图输出
    ov::Tensor anomaly_tensor = infer_request_.get_output_tensor(0);
    Mat anomaly_map(Size(metadata_.infer_size[1], metadata_.infer_size[0]), 
                   CV_32FC1, 
                   anomaly_tensor.data<float>());
    
    // 获取分数
    Mat pred_score;
    if (outputs_.size() == 2) {
        // 有专门的分数输出
        ov::Tensor score_tensor = infer_request_.get_output_tensor(1);
        pred_score = Mat(Size(1, 1), CV_32FC1, score_tensor.data<float>());
    } else {
        // 从异常图计算最大值作为分数
        double minVal, maxVal;
        minMaxLoc(anomaly_map, &minVal, &maxVal);
        float score_val = static_cast<float>(maxVal);
        pred_score = Mat(Size(1, 1), CV_32FC1, &score_val);
    }
    
    // 后处理：标准化和尺寸恢复
    vector<Mat> post_results = post_process(anomaly_map, pred_score, metadata_);
    
    return Result{post_results[0], post_results[1].at<float>(0, 0)};
}

Result PatchCoreDetector::detectSingle(Mat& image, float threshold) {
    auto start = chrono::high_resolution_clock::now();
    
    // 执行推理
    Result result = performInference(image);
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "[INFO] Detection completed in " << duration.count() << " ms" << endl;
    cout << "[INFO] Anomaly score: " << result.score << endl;
    
    // 生成可视化图像
    vector<Mat> vis_images = gen_images(image, result.anomaly_map, result.score, threshold);
    
    // 拼接所有可视化结果
    Mat combined_result;
    hconcat(vis_images, combined_result);
    
    return Result{combined_result, result.score};
}

double PatchCoreDetector::detectBatch(const string& image_dir, const string& save_dir, float threshold) {
    cout << "[INFO] Starting batch detection..." << endl;
    cout << "[INFO] Input directory: " << image_dir << endl;
    cout << "[INFO] Output directory: " << save_dir << endl;
    
    // 创建输出目录
    if (!filesystem::exists(save_dir)) {
        filesystem::create_directories(save_dir);
    }
    
    // 获取所有图像路径
    string img_dir = const_cast<string&>(image_dir);
    vector<String> image_paths = getImagePaths(img_dir);
    
    if (image_paths.empty()) {
        cout << "[WARNING] No images found in directory: " << image_dir << endl;
        return 0.0;
    }
    
    cout << "[INFO] Found " << image_paths.size() << " images" << endl;
    
    vector<double> inference_times;
    
    for (size_t i = 0; i < image_paths.size(); ++i) {
        cout << "[INFO] Processing (" << (i + 1) << "/" << image_paths.size() << "): " 
             << image_paths[i] << endl;
        
        // 读取图像
        string img_path = string(image_paths[i]);
        Mat image = readImage(img_path);
        
        if (image.empty()) {
            cout << "[WARNING] Failed to load image: " << image_paths[i] << endl;
            continue;
        }
        
        auto start = chrono::high_resolution_clock::now();
        
        // 执行推理
        Result result = performInference(image);
        
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        inference_times.push_back(duration.count());
        
        cout << "[INFO] Score: " << result.score << ", Time: " << duration.count() << " ms" << endl;
        
        // 生成可视化图像
        vector<Mat> vis_images = gen_images(image, result.anomaly_map, result.score, threshold);
        Mat combined_result;
        hconcat(vis_images, combined_result);
        
        // 保存结果
        String img_path_str = image_paths[i];
        string save_dir_str = const_cast<string&>(save_dir);
        saveScoreAndImages(result.score, combined_result, img_path_str, save_dir_str);
    }
    
    // 计算平均推理时间
    double avg_time = 0.0;
    if (!inference_times.empty()) {
        avg_time = accumulate(inference_times.begin(), inference_times.end(), 0.0) / inference_times.size();
    }
    
    cout << "[INFO] Batch detection completed!" << endl;
    cout << "[INFO] Average inference time: " << avg_time << " ms" << endl;
    
    return avg_time;
}

string PatchCoreDetector::getModelInfo() const {
    stringstream info;
    info << "[INFO] Model Information:" << endl;
    
    // 输入信息
    for (const auto& input : inputs_) {
        info << "  Input: " << input.get_any_name() << " [";
        for (size_t i = 0; i < input.get_shape().size(); ++i) {
            info << input.get_shape()[i];
            if (i < input.get_shape().size() - 1) info << ", ";
        }
        info << "] " << input.get_element_type() << endl;
    }
    
    // 输出信息
    for (const auto& output : outputs_) {
        info << "  Output: " << output.get_any_name() << " [";
        for (size_t i = 0; i < output.get_shape().size(); ++i) {
            info << output.get_shape()[i];
            if (i < output.get_shape().size() - 1) info << ", ";
        }
        info << "] " << output.get_element_type() << endl;
    }
    
    return info.str();
}

void PatchCoreDetector::setThresholds(float image_threshold, float pixel_threshold) {
    metadata_.image_threshold = image_threshold;
    metadata_.pixel_threshold = pixel_threshold;
    cout << "[INFO] Thresholds updated - Image: " << image_threshold 
         << ", Pixel: " << pixel_threshold << endl;
}

vector<string> PatchCoreDetector::getAvailableDevices() {
    vector<string> devices;
    try {
        ov::Core core;
        auto available_devices = core.get_available_devices();
        for (const auto& device : available_devices) {
            devices.push_back(device);
        }
    } catch (const exception& e) {
        cout << "[WARNING] Failed to get available devices: " << e.what() << endl;
    }
    return devices;
}