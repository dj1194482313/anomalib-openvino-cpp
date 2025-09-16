/**
 * @file usage_example.cpp
 * @brief Example usage of PatchCore detector
 */

#include "../patchcore_detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <chrono>

using namespace std;
using namespace cv;

int main() {
    try {
        // Model paths (update these to your actual model paths)
        string model_path = "model.xml";
        string metadata_path = "metadata.json";
        string test_image = "test.jpg";
        string output_dir = "./results";
        
        // Check if files exist
        if (!filesystem::exists(model_path)) {
            cerr << "[ERROR] Model file not found: " << model_path << endl;
            cerr << "[INFO] Please update the model_path variable with your actual model path" << endl;
            return 1;
        }
        
        if (!filesystem::exists(metadata_path)) {
            cerr << "[ERROR] Metadata file not found: " << metadata_path << endl;
            cerr << "[INFO] Please update the metadata_path variable with your actual metadata path" << endl;
            return 1;
        }
        
        // Create output directory
        filesystem::create_directories(output_dir);
        
        cout << "[INFO] Initializing PatchCore detector..." << endl;
        
        // Initialize detector
        PatchCoreDetector detector(
            model_path,     // Model file path
            metadata_path,  // Metadata file path
            "CPU",          // Device (CPU/GPU/AUTO)
            true,           // Use OpenVINO preprocessing
            false           // Is EfficientAD model
        );
        
        cout << "[INFO] Detector initialized successfully!" << endl;
        
        // Test 1: Single image detection
        if (filesystem::exists(test_image)) {
            cout << "\n[INFO] Testing single image detection..." << endl;
            
            Mat image = imread(test_image, IMREAD_COLOR);
            if (image.empty()) {
                cerr << "[ERROR] Failed to load test image: " << test_image << endl;
                return 1;
            }
            
            // Convert BGR to RGB
            cvtColor(image, image, COLOR_BGR2RGB);
            
            // Detect anomalies
            Result result = detector.detectSingle(image, 0.5f);
            
            cout << "[INFO] Detection result:" << endl;
            cout << "  Anomaly score: " << result.score << endl;
            cout << "  Result image size: " << result.anomaly_map.size() << endl;
            
            // Save result
            string output_file = output_dir + "/single_detection_result.jpg";
            imwrite(output_file, result.anomaly_map);
            cout << "[INFO] Result saved to: " << output_file << endl;
            
        } else {
            cout << "[WARNING] Test image not found: " << test_image << endl;
            cout << "[INFO] Creating a dummy test image..." << endl;
            
            // Create a dummy test image
            Mat dummy_image = Mat::zeros(224, 224, CV_8UC3);
            randu(dummy_image, Scalar(0, 0, 0), Scalar(255, 255, 255));
            
            Result result = detector.detectSingle(dummy_image, 0.5f);
            cout << "[INFO] Dummy detection result - Anomaly score: " << result.score << endl;
        }
        
        // Test 2: Performance benchmark
        cout << "\n[INFO] Running performance benchmark..." << endl;
        
        // Create test image for benchmarking
        Mat bench_image = Mat::zeros(224, 224, CV_8UC3);
        randu(bench_image, Scalar(0, 0, 0), Scalar(255, 255, 255));
        
        const int num_iterations = 10;
        vector<double> times;
        
        cout << "[INFO] Warming up..." << endl;
        for (int i = 0; i < 3; ++i) {
            detector.detectSingle(bench_image);
        }
        
        cout << "[INFO] Running " << num_iterations << " iterations..." << endl;
        for (int i = 0; i < num_iterations; ++i) {
            auto start = chrono::high_resolution_clock::now();
            detector.detectSingle(bench_image);
            auto end = chrono::high_resolution_clock::now();
            
            auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0); // Convert to milliseconds
            
            cout << "[INFO] Iteration " << (i + 1) << ": " << times.back() << " ms" << endl;
        }
        
        // Calculate statistics
        double mean_time = accumulate(times.begin(), times.end(), 0.0) / times.size();
        double fps = 1000.0 / mean_time;
        
        cout << "\n[INFO] Benchmark Results:" << endl;
        cout << "  Average time: " << fixed << setprecision(2) << mean_time << " ms" << endl;
        cout << "  Throughput: " << fps << " FPS" << endl;
        
        // Test 3: Device information
        cout << "\n[INFO] Available devices:" << endl;
        auto devices = PatchCoreDetector::getAvailableDevices();
        for (const auto& device : devices) {
            cout << "  - " << device << endl;
        }
        
        cout << "\n[SUCCESS] All tests completed successfully!" << endl;
        return 0;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception caught: " << e.what() << endl;
        return 1;
    }
}