# PatchCore OpenVINO C++ Deployment

A high-performance C++ implementation for anomaly detection using PatchCore models with OpenVINO inference engine.

## 🚀 Features

- **High Performance**: Optimized C++17 implementation with OpenVINO acceleration
- **Cross-Platform**: Supports Linux, Windows, and macOS
- **Flexible Deployment**: CPU and GPU inference support
- **Complete Pipeline**: Image preprocessing, inference, postprocessing, and visualization
- **Batch Processing**: Efficient batch inference for multiple images
- **Benchmarking**: Built-in performance measurement tools
- **Easy Integration**: Clean API design for easy integration into existing projects

## 📋 Requirements

### System Requirements
- **Operating System**: Linux (Ubuntu 18.04+), Windows 10+, or macOS 10.15+
- **Compiler**: GCC 7.3+, Clang 6.0+, or MSVC 2019+
- **CMake**: 3.19 or later

### Dependencies
- **OpenVINO**: 2023.0 or later
- **OpenCV**: 4.0 or later
- **C++ Standard**: C++17

## 🛠️ Installation

### 1. Install OpenVINO

#### Linux/macOS
```bash
# Download and install OpenVINO from Intel's official website
# https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html

# Source the environment
source /opt/intel/openvino_2023/setupvars.sh
```

#### Windows
```cmd
REM Download and install OpenVINO from Intel's official website
REM Run the setupvars.bat script
"C:\Program Files (x86)\Intel\openvino_2023\setupvars.bat"
```

### 2. Install OpenCV

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install libopencv-dev
```

#### macOS (using Homebrew)
```bash
brew install opencv
```

#### Windows
Download pre-built OpenCV from https://opencv.org/releases/

### 3. Build the Project

#### Using Build Scripts (Recommended)

**Linux/macOS:**
```bash
# Basic build
./build.sh

# Debug build with clean
./build.sh --type Debug --clean

# Custom paths
./build.sh --opencv-dir /usr/local --openvino-dir /opt/intel/openvino_2023
```

**Windows:**
```cmd
REM Basic build
build.bat

REM Debug build with clean
build.bat --type Debug --clean

REM Custom paths
build.bat --opencv-dir "C:\opencv" --openvino-dir "C:\Intel\openvino_2023"
```

#### Manual Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -j$(nproc)
```

## 🎯 Usage

### Command Line Interface

The main executable provides a comprehensive command-line interface:

```bash
# Show help
./main --help

# Single image detection
./main -m model.xml -t metadata.json -i test.jpg

# Batch processing
./main -m model.xml -t metadata.json -b ./test_images -o ./results

# Performance benchmark
./main -m model.xml -t metadata.json --benchmark -d GPU -n 100

# List available devices
./main --list-devices
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Path to OpenVINO model file (.xml) | Required |
| `-t, --metadata` | Path to metadata file (.json) | Required |
| `-i, --image` | Path to input image (single detection) | - |
| `-b, --batch` | Path to image directory (batch detection) | - |
| `-o, --output` | Output directory for results | `./results` |
| `-d, --device` | Inference device (CPU, GPU, AUTO) | `CPU` |
| `-th, --threshold` | Anomaly threshold [0.0-1.0] | `0.5` |
| `--openvino-preprocess` | Use OpenVINO preprocessing | `true` |
| `--efficient-ad` | EfficientAD model mode | `false` |
| `--benchmark` | Run performance benchmark | - |
| `-n, --iterations` | Benchmark iterations | `100` |
| `--no-display` | Don't display result images | - |

### Programming Interface

```cpp
#include "patchcore_detector.h"

// Initialize detector
PatchCoreDetector detector(
    "model.xml",           // Model path
    "metadata.json",       // Metadata path
    "CPU",                 // Device
    true,                  // Use OpenVINO preprocessing
    false                  // Is EfficientAD model
);

// Single image detection
cv::Mat image = cv::imread("test.jpg");
cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

Result result = detector.detectSingle(image, 0.5);
std::cout << "Anomaly score: " << result.score << std::endl;

// Save result
cv::imwrite("result.jpg", result.anomaly_map);

// Batch processing
double avg_time = detector.detectBatch("./images", "./results", 0.5);
std::cout << "Average time: " << avg_time << " ms" << std::endl;
```

## 📊 Model Export

### From Anomalib Checkpoint

Use the provided Python script to export models from Anomalib checkpoints:

```bash
python export_model_fixed.py \
    --config config.yaml \
    --checkpoint model.ckpt \
    --output ./exported_model
```

This script fixes common device mismatch issues when exporting models trained on CUDA to CPU-compatible formats.

### Export Script Features

- **Device Compatibility**: Automatically fixes CUDA/CPU tensor device mismatches
- **Complete Pipeline**: Exports both ONNX and OpenVINO IR formats
- **Metadata Generation**: Creates compatible metadata files for C++ inference
- **Error Recovery**: Robust error handling and validation

## 🎨 Visualization

The tool provides comprehensive visualization features:

1. **Anomaly Heatmap**: Color-coded anomaly intensity map
2. **Segmentation Mask**: Binary mask highlighting anomalous regions
3. **Masked Image**: Original image with anomalous regions highlighted
4. **Border Visualization**: Edge detection around anomalous regions
5. **Superimposed Result**: Heatmap overlaid on original image with confidence score

## 📈 Performance

### Benchmarking

Run performance benchmarks to evaluate inference speed:

```bash
# CPU benchmark
./main -m model.xml -t metadata.json --benchmark -d CPU -n 100

# GPU benchmark (if available)
./main -m model.xml -t metadata.json --benchmark -d GPU -n 100
```

### Optimization Tips

1. **Use GPU**: Enable GPU acceleration if available (`-d GPU`)
2. **OpenVINO Preprocessing**: Use built-in preprocessing for better performance
3. **Batch Processing**: Process multiple images in batch mode for efficiency
4. **Model Optimization**: Use OpenVINO Model Optimizer for additional speedup

## 🔧 Troubleshooting

### Common Issues

#### Build Errors

**OpenVINO not found:**
```bash
# Set OpenVINO environment
source /opt/intel/openvino_2023/setupvars.sh  # Linux/macOS
# or
"C:\Intel\openvino_2023\setupvars.bat"        # Windows
```

**OpenCV not found:**
```bash
# Specify OpenCV path
./build.sh --opencv-dir /usr/local/lib/cmake/opencv4
```

#### Runtime Errors

**Model loading failed:**
- Verify model file paths are correct
- Check if metadata.json exists and is properly formatted
- Ensure model was exported correctly from Anomalib

**Device not supported:**
- Use `--list-devices` to see available devices
- Fall back to CPU if GPU is not available

**Memory issues:**
- Reduce batch size for large images
- Use CPU device if GPU memory is insufficient

### Debug Mode

Build in Debug mode for detailed error information:

```bash
./build.sh --type Debug
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow C++17 best practices
2. **Documentation**: Update documentation for new features
3. **Testing**: Add tests for new functionality
4. **Compatibility**: Ensure cross-platform compatibility

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **Intel OpenVINO**: High-performance inference engine
- **OpenCV**: Computer vision library
- **Anomalib**: Original PatchCore implementation
- **RapidJSON**: Fast JSON parsing library

## 📞 Support

For issues and questions:

1. Check the [troubleshooting section](#-troubleshooting)
2. Search existing GitHub issues
3. Create a new issue with detailed information

## 🔄 Version History

### v1.0.0
- Initial release with PatchCore support
- Cross-platform build system
- Comprehensive visualization
- Performance benchmarking
- Model export script with device compatibility fixes
