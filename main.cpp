#include "patchcore_detector.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace std;
using namespace cv;

/**
 * @brief 显示使用帮助
 */
void showHelp() {
    cout << "PatchCore OpenVINO C++ Deployment Tool\n" << endl;
    cout << "Usage:" << endl;
    cout << "  Single image detection:" << endl;
    cout << "    ./main -m <model.xml> -t <metadata.json> -i <image.jpg> [-o <output_dir>] [-d <device>] [-th <threshold>]" << endl;
    cout << "  Batch detection:" << endl;
    cout << "    ./main -m <model.xml> -t <metadata.json> -b <image_dir> -o <output_dir> [-d <device>] [-th <threshold>]" << endl;
    cout << "  Performance benchmark:" << endl;
    cout << "    ./main -m <model.xml> -t <metadata.json> --benchmark [-d <device>] [-n <iterations>]" << endl;
    cout << "\nOptions:" << endl;
    cout << "  -m, --model <path>      Path to OpenVINO model file (.xml)" << endl;
    cout << "  -t, --metadata <path>   Path to metadata file (.json)" << endl;
    cout << "  -i, --image <path>      Path to input image for single detection" << endl;
    cout << "  -b, --batch <path>      Path to directory containing images for batch detection" << endl;
    cout << "  -o, --output <path>     Output directory for results (default: ./results)" << endl;
    cout << "  -d, --device <device>   Inference device: CPU, GPU, AUTO (default: CPU)" << endl;
    cout << "  -th, --threshold <val>  Anomaly threshold [0.0-1.0] (default: 0.5)" << endl;
    cout << "  --openvino-preprocess   Use OpenVINO preprocessing (default: true)" << endl;
    cout << "  --efficient-ad          Use EfficientAD model mode (default: false)" << endl;
    cout << "  --benchmark             Run performance benchmark" << endl;
    cout << "  -n, --iterations <num>  Number of benchmark iterations (default: 100)" << endl;
    cout << "  --no-display            Don't display result images" << endl;
    cout << "  --list-devices          List available devices and exit" << endl;
    cout << "  -h, --help              Show this help message" << endl;
    cout << "\nExamples:" << endl;
    cout << "  ./main -m model.xml -t metadata.json -i test.jpg" << endl;
    cout << "  ./main -m model.xml -t metadata.json -b ./test_images -o ./results" << endl;
    cout << "  ./main -m model.xml -t metadata.json --benchmark -d GPU -n 50" << endl;
}

/**
 * @brief 解析命令行参数
 */
struct Arguments {
    string model_path;
    string metadata_path;
    string image_path;
    string batch_dir;
    string output_dir = "./results";
    string device = "CPU";
    float threshold = 0.5f;
    bool use_openvino_preprocess = true;
    bool is_efficient_ad = false;
    bool benchmark_mode = false;
    bool single_mode = false;
    bool batch_mode = false;
    bool no_display = false;
    bool list_devices = false;
    int benchmark_iterations = 100;
};

Arguments parseArguments(int argc, char* argv[]) {
    Arguments args;
    
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            showHelp();
            exit(0);
        } else if (arg == "--list-devices") {
            args.list_devices = true;
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) args.model_path = argv[++i];
        } else if (arg == "-t" || arg == "--metadata") {
            if (i + 1 < argc) args.metadata_path = argv[++i];
        } else if (arg == "-i" || arg == "--image") {
            if (i + 1 < argc) {
                args.image_path = argv[++i];
                args.single_mode = true;
            }
        } else if (arg == "-b" || arg == "--batch") {
            if (i + 1 < argc) {
                args.batch_dir = argv[++i];
                args.batch_mode = true;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) args.output_dir = argv[++i];
        } else if (arg == "-d" || arg == "--device") {
            if (i + 1 < argc) args.device = argv[++i];
        } else if (arg == "-th" || arg == "--threshold") {
            if (i + 1 < argc) args.threshold = stof(argv[++i]);
        } else if (arg == "--openvino-preprocess") {
            args.use_openvino_preprocess = true;
        } else if (arg == "--efficient-ad") {
            args.is_efficient_ad = true;
        } else if (arg == "--benchmark") {
            args.benchmark_mode = true;
        } else if (arg == "-n" || arg == "--iterations") {
            if (i + 1 < argc) args.benchmark_iterations = stoi(argv[++i]);
        } else if (arg == "--no-display") {
            args.no_display = true;
        }
    }
    
    return args;
}

/**
 * @brief 验证参数
 */
bool validateArguments(const Arguments& args) {
    if (args.list_devices) return true;
    
    if (args.model_path.empty()) {
        cerr << "[ERROR] Model path is required (-m/--model)" << endl;
        return false;
    }
    
    if (args.metadata_path.empty()) {
        cerr << "[ERROR] Metadata path is required (-t/--metadata)" << endl;
        return false;
    }
    
    if (!filesystem::exists(args.model_path)) {
        cerr << "[ERROR] Model file not found: " << args.model_path << endl;
        return false;
    }
    
    if (!filesystem::exists(args.metadata_path)) {
        cerr << "[ERROR] Metadata file not found: " << args.metadata_path << endl;
        return false;
    }
    
    if (!args.benchmark_mode && !args.single_mode && !args.batch_mode) {
        cerr << "[ERROR] Must specify either single image (-i), batch directory (-b), or benchmark mode" << endl;
        return false;
    }
    
    if (args.single_mode && !filesystem::exists(args.image_path)) {
        cerr << "[ERROR] Image file not found: " << args.image_path << endl;
        return false;
    }
    
    if (args.batch_mode && !filesystem::exists(args.batch_dir)) {
        cerr << "[ERROR] Batch directory not found: " << args.batch_dir << endl;
        return false;
    }
    
    if (args.threshold < 0.0f || args.threshold > 1.0f) {
        cerr << "[ERROR] Threshold must be between 0.0 and 1.0" << endl;
        return false;
    }
    
    return true;
}

/**
 * @brief 运行性能基准测试
 */
void runBenchmark(PatchCoreDetector& detector, int iterations) {
    cout << "\n[INFO] Starting performance benchmark..." << endl;
    cout << "[INFO] Iterations: " << iterations << endl;
    
    // 创建测试图像
    Mat test_image = Mat::zeros(224, 224, CV_8UC3);
    randu(test_image, Scalar(0, 0, 0), Scalar(255, 255, 255));
    
    vector<double> times;
    times.reserve(iterations);
    
    cout << "[INFO] Warming up..." << endl;
    for (int i = 0; i < 10; ++i) {
        detector.detectSingle(test_image);
    }
    
    cout << "[INFO] Running benchmark..." << endl;
    for (int i = 0; i < iterations; ++i) {
        auto start = chrono::high_resolution_clock::now();
        detector.detectSingle(test_image);
        auto end = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0); // Convert to milliseconds
        
        if ((i + 1) % 10 == 0) {
            cout << "[INFO] Completed " << (i + 1) << "/" << iterations << " iterations" << endl;
        }
    }
    
    // 计算统计信息
    sort(times.begin(), times.end());
    double mean = accumulate(times.begin(), times.end(), 0.0) / times.size();
    double median = times[times.size() / 2];
    double min_time = times.front();
    double max_time = times.back();
    double p95 = times[static_cast<size_t>(times.size() * 0.95)];
    
    cout << "\n[INFO] Benchmark Results:" << endl;
    cout << "  Mean:   " << fixed << setprecision(2) << mean << " ms" << endl;
    cout << "  Median: " << median << " ms" << endl;
    cout << "  Min:    " << min_time << " ms" << endl;
    cout << "  Max:    " << max_time << " ms" << endl;
    cout << "  P95:    " << p95 << " ms" << endl;
    cout << "  FPS:    " << (1000.0 / mean) << endl;
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    try {
        // 解析命令行参数
        Arguments args = parseArguments(argc, argv);
        
        // 列出可用设备
        if (args.list_devices) {
            cout << "Available devices:" << endl;
            auto devices = PatchCoreDetector::getAvailableDevices();
            for (const auto& device : devices) {
                cout << "  " << device << endl;
            }
            return 0;
        }
        
        // 验证参数
        if (!validateArguments(args)) {
            showHelp();
            return 1;
        }
        
        // 创建输出目录
        if (!args.benchmark_mode) {
            filesystem::create_directories(args.output_dir);
        }
        
        // 初始化检测器
        cout << "[INFO] Initializing PatchCore detector..." << endl;
        PatchCoreDetector detector(
            args.model_path,
            args.metadata_path,
            args.device,
            args.use_openvino_preprocess,
            args.is_efficient_ad
        );
        
        // 执行相应的模式
        if (args.benchmark_mode) {
            runBenchmark(detector, args.benchmark_iterations);
        } else if (args.single_mode) {
            cout << "\n[INFO] Processing single image: " << args.image_path << endl;
            
            Mat image = imread(args.image_path, IMREAD_COLOR);
            if (image.empty()) {
                cerr << "[ERROR] Failed to load image: " << args.image_path << endl;
                return 1;
            }
            
            // Convert BGR to RGB
            cvtColor(image, image, COLOR_BGR2RGB);
            
            // 执行检测
            Result result = detector.detectSingle(image, args.threshold);
            
            // 保存结果
            string filename = filesystem::path(args.image_path).stem().string();
            string output_path = args.output_dir + "/" + filename + "_result.jpg";
            imwrite(output_path, result.anomaly_map);
            
            cout << "[INFO] Result saved to: " << output_path << endl;
            
            // 显示结果
            if (!args.no_display) {
                namedWindow("Detection Result", WINDOW_NORMAL);
                resizeWindow("Detection Result", 1200, 400);
                imshow("Detection Result", result.anomaly_map);
                cout << "[INFO] Press any key to exit..." << endl;
                waitKey(0);
                destroyAllWindows();
            }
            
        } else if (args.batch_mode) {
            cout << "\n[INFO] Processing batch directory: " << args.batch_dir << endl;
            
            double avg_time = detector.detectBatch(args.batch_dir, args.output_dir, args.threshold);
            
            cout << "\n[INFO] Batch processing completed!" << endl;
            cout << "[INFO] Average inference time: " << avg_time << " ms" << endl;
            cout << "[INFO] Results saved to: " << args.output_dir << endl;
        }
        
        cout << "\n[INFO] Processing completed successfully!" << endl;
        return 0;
        
    } catch (const exception& e) {
        cerr << "[ERROR] " << e.what() << endl;
        return 1;
    }
}
