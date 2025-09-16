# Examples

This directory contains example code and configuration files for using the PatchCore OpenVINO C++ implementation.

## Files

### Configuration Examples

- **`metadata_example.json`**: Example metadata file format compatible with the C++ implementation
  - Contains thresholds, normalization parameters, and image dimensions
  - Generated automatically by the export script or can be created manually

### Code Examples

- **`usage_example.cpp`**: Complete example showing how to use the PatchCoreDetector class
  - Single image detection
  - Performance benchmarking
  - Device enumeration
  - Proper error handling

## Usage

### Running the Usage Example

```bash
# Build the main project first
./build.sh

# Then you can compile and run the usage example
cd examples
g++ -std=c++17 -I.. usage_example.cpp ../build/libanomab_core.a $(pkg-config --cflags --libs opencv4) -lopenvino -o usage_example
./usage_example
```

### Using the Metadata Example

The `metadata_example.json` file shows the required format for model metadata:

```json
{
  "image_threshold": 0.5,    // Image-level anomaly threshold
  "pixel_threshold": 0.5,    // Pixel-level anomaly threshold  
  "min": 0.0,                // Minimum normalization value
  "max": 1.0,                // Maximum normalization value
  "transform": {
    "transform": {
      "transforms": [
        {
          "height": 224,       // Model input height
          "width": 224         // Model input width
        }
      ]
    }
  }
}
```

### Model Export Example

```python
# Export a model from Anomalib checkpoint
python export_model_fixed.py \
    --config path/to/config.yaml \
    --checkpoint path/to/model.ckpt \
    --output ./exported_model

# This will generate:
# - model.xml (OpenVINO IR model)
# - model.bin (OpenVINO IR weights)
# - metadata.json (Model metadata)
```

### Complete Workflow Example

```bash
# 1. Export model from Anomalib
python export_model_fixed.py -c config.yaml -k model.ckpt -o ./model

# 2. Build the C++ project
./build.sh

# 3. Run single image detection
./build/main -m ./model/model.xml -t ./model/metadata.json -i test.jpg

# 4. Run batch processing
./build/main -m ./model/model.xml -t ./model/metadata.json -b ./test_images -o ./results

# 5. Run performance benchmark
./build/main -m ./model/model.xml -t ./model/metadata.json --benchmark -n 100
```

## Notes

- Update file paths in the examples to match your actual model and image locations
- Ensure OpenVINO and OpenCV are properly installed and configured
- The examples assume RGB input images (the detector handles BGR to RGB conversion internally)
- For EfficientAD models, use the `--efficient-ad` flag or set the parameter in the constructor