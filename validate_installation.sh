#!/bin/bash

# PatchCore OpenVINO C++ Installation Validation Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info "PatchCore OpenVINO C++ Installation Validation"
print_info "=============================================="

# Check if we're in the project root
if [[ ! -f "CMakeLists.txt" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Test 1: Check for required tools
print_info "\n1. Checking required tools..."

# Check CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "CMake found: $CMAKE_VERSION"
else
    print_error "CMake not found. Please install CMake 3.19 or later."
    exit 1
fi

# Check C++ compiler
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "GCC found: $GCC_VERSION"
elif command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "Clang found: $CLANG_VERSION"
else
    print_error "C++ compiler not found. Please install GCC or Clang."
    exit 1
fi

# Check Python (for export script)
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "Python found: $PYTHON_VERSION"
else
    print_warning "Python3 not found. Model export script will not work."
fi

# Test 2: Check dependencies
print_info "\n2. Checking dependencies..."

# Check OpenCV
if pkg-config --exists opencv4; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    print_success "OpenCV found: $OPENCV_VERSION"
    OPENCV_OK=true
elif pkg-config --exists opencv; then
    OPENCV_VERSION=$(pkg-config --modversion opencv)
    print_success "OpenCV found: $OPENCV_VERSION"
    OPENCV_OK=true
else
    print_error "OpenCV not found via pkg-config"
    print_info "Please install OpenCV or specify --opencv-dir when building"
    OPENCV_OK=false
fi

# Check OpenVINO
if [[ -n "$INTEL_OPENVINO_DIR" ]]; then
    print_success "OpenVINO environment found: $INTEL_OPENVINO_DIR"
    OPENVINO_OK=true
elif [[ -f "/opt/intel/openvino/setupvars.sh" ]]; then
    print_warning "OpenVINO found but not sourced. Please run:"
    print_info "source /opt/intel/openvino/setupvars.sh"
    OPENVINO_OK=false
else
    print_error "OpenVINO not found"
    print_info "Please install OpenVINO and source setupvars.sh"
    OPENVINO_OK=false
fi

# Test 3: Validate project structure
print_info "\n3. Validating project structure..."

REQUIRED_FILES=(
    "CMakeLists.txt"
    "patchcore_detector.h"
    "patchcore_detector.cpp"
    "main.cpp"
    "utils.h"
    "utils.cpp"
    "opencv_utils.h"
    "opencv_utils.cpp"
    "inference.hpp"
    "export_model_fixed.py"
    "build.sh"
    "README.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        print_success "Found: $file"
    else
        print_error "Missing: $file"
        exit 1
    fi
done

# Test 4: Test CMake configuration
print_info "\n4. Testing CMake configuration..."

mkdir -p validation_build
cd validation_build

if [[ "$OPENCV_OK" == true && "$OPENVINO_OK" == true ]]; then
    print_info "Attempting full CMake configuration..."
    if cmake -DCMAKE_BUILD_TYPE=Release .. > cmake_output.log 2>&1; then
        print_success "CMake configuration successful"
        
        # Try to build
        print_info "Attempting to build project..."
        if cmake --build . --config Release -j$(nproc) > build_output.log 2>&1; then
            print_success "Build successful!"
            
            # Check if executable was created
            if [[ -f "main" ]] || [[ -f "Release/main" ]] || [[ -f "main.exe" ]] || [[ -f "Release/main.exe" ]]; then
                print_success "Executable created successfully"
                
                # Try to run help
                if [[ -f "main" ]]; then
                    MAIN_EXEC="./main"
                elif [[ -f "Release/main" ]]; then
                    MAIN_EXEC="./Release/main"
                elif [[ -f "main.exe" ]]; then
                    MAIN_EXEC="./main.exe"
                elif [[ -f "Release/main.exe" ]]; then
                    MAIN_EXEC="./Release/main.exe"
                fi
                
                print_info "Testing executable..."
                if $MAIN_EXEC --help > help_output.log 2>&1; then
                    print_success "Executable runs correctly"
                else
                    print_warning "Executable created but help command failed"
                    print_info "This might be due to missing model files, which is expected"
                fi
            else
                print_warning "Build completed but executable not found"
            fi
        else
            print_error "Build failed. Check validation_build/build_output.log"
            tail -10 build_output.log
        fi
    else
        print_error "CMake configuration failed. Check validation_build/cmake_output.log"
        tail -10 cmake_output.log
    fi
else
    print_warning "Skipping build test due to missing dependencies"
fi

cd ..

# Test 5: Validate Python export script
print_info "\n5. Validating Python export script..."

if command -v python3 &> /dev/null; then
    if python3 -c "import torch" 2>/dev/null; then
        print_success "PyTorch available"
    else
        print_warning "PyTorch not available - export script may not work"
    fi
    
    if python3 -c "import openvino" 2>/dev/null; then
        print_success "OpenVINO Python API available"
    else
        print_warning "OpenVINO Python API not available - export script may not work"
    fi
    
    if python3 export_model_fixed.py --help > /dev/null 2>&1; then
        print_success "Export script syntax is valid"
    else
        print_warning "Export script has syntax issues"
    fi
else
    print_warning "Python3 not available - skipping export script validation"
fi

# Summary
print_info "\n6. Validation Summary"
print_info "===================="

if [[ "$OPENCV_OK" == true && "$OPENVINO_OK" == true ]]; then
    print_success "All major dependencies are available"
    print_info "You should be able to build and use the project"
else
    print_warning "Some dependencies are missing:"
    if [[ "$OPENCV_OK" != true ]]; then
        print_info "  - Install OpenCV development packages"
    fi
    if [[ "$OPENVINO_OK" != true ]]; then
        print_info "  - Install OpenVINO and source setupvars.sh"
    fi
fi

print_info "\nNext steps:"
print_info "1. If dependencies are available, run: ./build.sh"
print_info "2. Export a model using: python export_model_fixed.py --help"
print_info "3. Run detection: ./build/main --help"

# Cleanup
if [[ -d "validation_build" ]]; then
    print_info "\nCleaning up validation build directory..."
    rm -rf validation_build
fi

print_info "\nValidation completed!"