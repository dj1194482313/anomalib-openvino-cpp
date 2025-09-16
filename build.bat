@echo off
REM PatchCore OpenVINO C++ Build Script for Windows
REM Supports Visual Studio 2019 and 2022

setlocal enabledelayedexpansion

REM Default values
set BUILD_TYPE=Release
set BUILD_DIR=build
set INSTALL_PREFIX=
set OPENCV_DIR=
set OPENVINO_DIR=
set CLEAN_BUILD=false
set VERBOSE=false
set GENERATOR="Visual Studio 16 2019"
set PLATFORM=x64

REM Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-t" (
    set BUILD_TYPE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--type" (
    set BUILD_TYPE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-d" (
    set BUILD_DIR=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--dir" (
    set BUILD_DIR=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-p" (
    set INSTALL_PREFIX=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--prefix" (
    set INSTALL_PREFIX=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--opencv-dir" (
    set OPENCV_DIR=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--openvino-dir" (
    set OPENVINO_DIR=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-c" (
    set CLEAN_BUILD=true
    shift
    goto parse_args
)
if /i "%~1"=="--clean" (
    set CLEAN_BUILD=true
    shift
    goto parse_args
)
if /i "%~1"=="-v" (
    set VERBOSE=true
    shift
    goto parse_args
)
if /i "%~1"=="--verbose" (
    set VERBOSE=true
    shift
    goto parse_args
)
if /i "%~1"=="--vs2019" (
    set GENERATOR="Visual Studio 16 2019"
    shift
    goto parse_args
)
if /i "%~1"=="--vs2022" (
    set GENERATOR="Visual Studio 17 2022"
    shift
    goto parse_args
)
echo [ERROR] Unknown option: %~1
exit /b 1

:show_help
echo PatchCore OpenVINO C++ Build Script for Windows
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   -t, --type ^<type^>       Build type: Debug, Release, RelWithDebInfo (default: Release)
echo   -d, --dir ^<dir^>         Build directory (default: build)
echo   -p, --prefix ^<path^>     Install prefix
echo   --opencv-dir ^<path^>     OpenCV installation directory
echo   --openvino-dir ^<path^>   OpenVINO installation directory
echo   -c, --clean             Clean build (remove build directory first)
echo   -v, --verbose           Verbose build output
echo   --vs2019                Use Visual Studio 2019 generator
echo   --vs2022                Use Visual Studio 2022 generator (default)
echo   -h, --help              Show this help message
echo.
echo Examples:
echo   %~nx0                                     # Basic build
echo   %~nx0 -t Debug -c                        # Clean debug build
echo   %~nx0 --opencv-dir C:\opencv              # Specify OpenCV path
echo   %~nx0 --openvino-dir C:\intel\openvino   # Specify OpenVINO path
exit /b 0

:end_parse

echo [INFO] Starting PatchCore OpenVINO C++ build process...
echo [INFO] Build type: %BUILD_TYPE%
echo [INFO] Build directory: %BUILD_DIR%
echo [INFO] Generator: %GENERATOR%

REM Check if we're in the project root
if not exist "CMakeLists.txt" (
    echo [ERROR] CMakeLists.txt not found. Please run this script from the project root directory.
    exit /b 1
)

REM Clean build if requested
if /i "%CLEAN_BUILD%"=="true" (
    echo [INFO] Cleaning build directory...
    if exist "%BUILD_DIR%" (
        rmdir /s /q "%BUILD_DIR%"
    )
)

REM Create build directory
if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)

cd "%BUILD_DIR%"

REM Prepare CMake arguments
set CMAKE_ARGS=-G %GENERATOR% -A %PLATFORM%
set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

if not "%INSTALL_PREFIX%"=="" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%"
)

if not "%OPENCV_DIR%"=="" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DOpenCV_DIR="%OPENCV_DIR%"
)

if not "%OPENVINO_DIR%"=="" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DOpenVINO_DIR="%OPENVINO_DIR%"
)

REM Check for dependencies
echo [INFO] Checking dependencies...

REM Check for cmake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake is not installed. Please install CMake 3.19 or later.
    exit /b 1
)

for /f "tokens=3" %%v in ('cmake --version ^| findstr /r "[0-9]\.[0-9]\.[0-9]"') do (
    echo [INFO] Found CMake version: %%v
    goto cmake_found
)
:cmake_found

REM Try to find OpenVINO
if "%OPENVINO_DIR%"=="" (
    if not "%INTEL_OPENVINO_DIR%"=="" (
        echo [INFO] Found OpenVINO environment: %INTEL_OPENVINO_DIR%
        set CMAKE_ARGS=%CMAKE_ARGS% -DOpenVINO_DIR="%INTEL_OPENVINO_DIR%\runtime\cmake"
    ) else (
        echo [WARNING] OpenVINO not found. You may need to specify --openvino-dir or run setupvars.bat
    )
)

REM Configure
echo [INFO] Configuring project...
echo [INFO] CMake command: cmake %CMAKE_ARGS% ..

if /i "%VERBOSE%"=="true" (
    cmake %CMAKE_ARGS% ..
) else (
    cmake %CMAKE_ARGS% .. > cmake_config.log 2>&1
    if errorlevel 1 (
        echo [ERROR] CMake configuration failed. Check cmake_config.log for details.
        type cmake_config.log | more
        exit /b 1
    )
)

echo [SUCCESS] Configuration completed successfully!

REM Build
echo [INFO] Building project...

set BUILD_ARGS=--build . --config %BUILD_TYPE%

if /i "%VERBOSE%"=="true" (
    set BUILD_ARGS=%BUILD_ARGS% --verbose
)

if /i "%VERBOSE%"=="true" (
    cmake %BUILD_ARGS%
) else (
    cmake %BUILD_ARGS% > build.log 2>&1
    if errorlevel 1 (
        echo [ERROR] Build failed. Check build.log for details.
        type build.log | more
        exit /b 1
    )
)

echo [SUCCESS] Build completed successfully!

REM Check if main executable was created
set MAIN_EXEC=
if exist "%BUILD_TYPE%\main.exe" (
    set MAIN_EXEC=%BUILD_TYPE%\main.exe
) else if exist "main.exe" (
    set MAIN_EXEC=main.exe
)

if not "%MAIN_EXEC%"=="" (
    echo [SUCCESS] Executable 'main.exe' created successfully!
    echo [INFO] Executable location: %CD%\%MAIN_EXEC%
    echo [INFO] To run the program:
    echo   cd %BUILD_DIR%
    echo   %MAIN_EXEC% --help
) else (
    echo [WARNING] Main executable not found. Build may have failed.
)

echo [INFO] Build process completed!
echo [INFO] Build artifacts are in: %CD%

cd ..