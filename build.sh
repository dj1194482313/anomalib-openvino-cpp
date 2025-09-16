#!/bin/bash

# PatchCore OpenVINO C++ Build Script
# Supports Linux and macOS

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build"
INSTALL_PREFIX=""
OPENCV_DIR=""
OPENVINO_DIR=""
CLEAN_BUILD=false
VERBOSE=false
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "PatchCore OpenVINO C++ Build Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -t, --type <type>       Build type: Debug, Release, RelWithDebInfo (default: Release)"
            echo "  -d, --dir <dir>         Build directory (default: build)"
            echo "  -p, --prefix <path>     Install prefix"
            echo "  --opencv-dir <path>     OpenCV installation directory"
            echo "  --openvino-dir <path>   OpenVINO installation directory"
            echo "  -c, --clean             Clean build (remove build directory first)"
            echo "  -j, --jobs <num>        Number of parallel build jobs (default: auto-detect)"
            echo "  -v, --verbose           Verbose build output"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Basic build"
            echo "  $0 -t Debug -c                       # Clean debug build"
            echo "  $0 --opencv-dir /opt/opencv           # Specify OpenCV path"
            echo "  $0 --openvino-dir /opt/intel/openvino # Specify OpenVINO path"
            exit 0
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -d|--dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -p|--prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --opencv-dir)
            OPENCV_DIR="$2"
            shift 2
            ;;
        --openvino-dir)
            OPENVINO_DIR="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_info "Starting PatchCore OpenVINO C++ build process..."
print_info "Build type: $BUILD_TYPE"
print_info "Build directory: $BUILD_DIR"
print_info "Parallel jobs: $JOBS"

# Check if we're in the project root
if [[ ! -f "CMakeLists.txt" ]]; then
    print_error "CMakeLists.txt not found. Please run this script from the project root directory."
    exit 1
fi

# Clean build if requested
if [[ "$CLEAN_BUILD" == true ]]; then
    print_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Prepare CMake arguments
CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=$BUILD_TYPE")

if [[ -n "$INSTALL_PREFIX" ]]; then
    CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX")
fi

if [[ -n "$OPENCV_DIR" ]]; then
    CMAKE_ARGS+=("-DOpenCV_DIR=$OPENCV_DIR")
fi

if [[ -n "$OPENVINO_DIR" ]]; then
    CMAKE_ARGS+=("-DOpenVINO_DIR=$OPENVINO_DIR")
fi

# Check for dependencies
print_info "Checking dependencies..."

# Check for cmake
if ! command -v cmake &> /dev/null; then
    print_error "CMake is not installed. Please install CMake 3.19 or later."
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
print_info "Found CMake version: $CMAKE_VERSION"

# Try to find OpenCV
if [[ -z "$OPENCV_DIR" ]]; then
    if pkg-config --exists opencv4; then
        print_info "Found OpenCV via pkg-config"
    elif pkg-config --exists opencv; then
        print_info "Found OpenCV via pkg-config"
    else
        print_warning "OpenCV not found via pkg-config. You may need to specify --opencv-dir"
    fi
fi

# Try to find OpenVINO
if [[ -z "$OPENVINO_DIR" ]]; then
    if [[ -n "$INTEL_OPENVINO_DIR" ]]; then
        print_info "Found OpenVINO environment: $INTEL_OPENVINO_DIR"
        CMAKE_ARGS+=("-DOpenVINO_DIR=$INTEL_OPENVINO_DIR/runtime/cmake")
    else
        print_warning "OpenVINO not found. You may need to specify --openvino-dir or source setupvars.sh"
    fi
fi

# Configure
print_info "Configuring project..."
print_info "CMake command: cmake ${CMAKE_ARGS[*]} .."

if [[ "$VERBOSE" == true ]]; then
    cmake "${CMAKE_ARGS[@]}" ..
else
    cmake "${CMAKE_ARGS[@]}" .. > cmake_config.log 2>&1 || {
        print_error "CMake configuration failed. Check cmake_config.log for details."
        tail -20 cmake_config.log
        exit 1
    }
fi

print_success "Configuration completed successfully!"

# Build
print_info "Building project with $JOBS parallel jobs..."

BUILD_ARGS=()
BUILD_ARGS+=("--build" ".")
BUILD_ARGS+=("--config" "$BUILD_TYPE")
BUILD_ARGS+=("-j" "$JOBS")

if [[ "$VERBOSE" == true ]]; then
    BUILD_ARGS+=("--verbose")
fi

if [[ "$VERBOSE" == true ]]; then
    cmake "${BUILD_ARGS[@]}"
else
    cmake "${BUILD_ARGS[@]}" > build.log 2>&1 || {
        print_error "Build failed. Check build.log for details."
        tail -20 build.log
        exit 1
    }
fi

print_success "Build completed successfully!"

# Check if main executable was created
if [[ -f "main" ]] || [[ -f "$BUILD_TYPE/main" ]] || [[ -f "main.exe" ]] || [[ -f "$BUILD_TYPE/main.exe" ]]; then
    print_success "Executable 'main' created successfully!"
    
    # Find the executable
    MAIN_EXEC=""
    for possible_path in "main" "$BUILD_TYPE/main" "main.exe" "$BUILD_TYPE/main.exe"; do
        if [[ -f "$possible_path" ]]; then
            MAIN_EXEC="$possible_path"
            break
        fi
    done
    
    print_info "Executable location: $PWD/$MAIN_EXEC"
    
    # Show usage
    print_info "To run the program:"
    echo "  cd $BUILD_DIR"
    echo "  ./$MAIN_EXEC --help"
    
else
    print_warning "Main executable not found. Build may have failed."
fi

print_info "Build process completed!"
print_info "Build artifacts are in: $PWD"