# EdgeElite NPU Deployment Guide
## Snapdragon X-Elite NPU Acceleration

This guide explains how to deploy EdgeElite with full NPU acceleration on Snapdragon X-Elite devices.

## Overview

EdgeElite supports Qualcomm Snapdragon X-Elite NPU acceleration for:
- **ASR (Speech Recognition)**: Whisper models with QNN optimization
- **LLM (Language Models)**: Flan-T5 models with NPU acceleration  
- **OCR (Image Recognition)**: Text extraction with NPU acceleration

## Prerequisites

### Hardware Requirements
- Snapdragon X-Elite development device or laptop
- Linux environment (Ubuntu 20.04+ recommended)
- At least 8GB RAM, 16GB recommended

### Software Requirements
- Python 3.10+
- ONNX Runtime with QNN support
- Qualcomm QNN SDK (if building from source)

## Step 1: Environment Setup

### Install ONNX Runtime with QNN Support

**Option A: Pre-built wheel (Recommended)**
```bash
# Install onnxruntime-qnn for Linux
pip install onnxruntime-qnn

# Verify QNN provider is available
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should show: ['QNNExecutionProvider', 'CPUExecutionProvider']
```

**Option B: Build from source**
```bash
# Clone ONNX Runtime with QNN support
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# Build with QNN support
./build.sh --enable_qnn --qnn_home /path/to/qnn/sdk

# Install the built wheel
pip install build/Linux/Release/dist/onnxruntime-*.whl
```

## Step 2: Model Preparation

### Convert Whisper Models for NPU

The current INT8 models use `ConvInteger` ops which aren't supported by QNN EP. Convert them to `QOperator` format:

```bash
cd backend
python convert_whisper_for_npu.py
```

This creates:
- `whisper-small-onnx/onnx/encoder_model_npu.onnx`
- `whisper-small-onnx/onnx/decoder_model_npu.onnx`

### Verify Model Compatibility

```bash
python -c "
import onnx
model = onnx.load('whisper-small-onnx/onnx/encoder_model_npu.onnx')
conv_integer_ops = [op for op in model.graph.node if op.op_type == 'ConvInteger']
qlinear_conv_ops = [op for op in model.graph.node if op.op_type == 'QLinearConv']
print(f'ConvInteger ops: {len(conv_integer_ops)}')
print(f'QLinearConv ops: {len(qlinear_conv_ops)}')
"
```

Should show: `ConvInteger ops: 0, QLinearConv ops: >0`

## Step 3: Deploy to Device

### Copy Application
```bash
# Copy EdgeElite to device
scp -r edgeelite/ user@device:/home/user/
ssh user@device

# Install dependencies
cd edgeelite/backend
pip install -r requirements.txt
pip install onnxruntime-qnn
```

### Test NPU Availability
```bash
python -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
print('QNN available:', 'QNNExecutionProvider' in ort.get_available_providers())
"
```

## Step 4: Run with NPU Acceleration

### Start Backend Server
```bash
cd backend
python main.py
```

Expected output:
```
🚀 Starting EdgeElite Backend Server...
🎤 ASR: QNN NPU optimized Whisper
🚀 Trying to load NPU-optimized encoder...
✅ Encoder loaded with provider: QNNExecutionProvider (NPU-optimized)
🚀 Trying to load NPU-optimized decoder...
✅ Decoder loaded with provider: QNNExecutionProvider (NPU-optimized)
```

### Test ASR with NPU
```bash
# Record audio and test transcription
curl -X POST http://localhost:8000/asr
```

## Step 5: Performance Monitoring

### Check NPU Utilization
```bash
# Monitor NPU usage (device-specific)
cat /sys/kernel/debug/qnn/htp/status
```

### Profile Performance
```bash
# ONNX Runtime profiling is enabled by default
# Check profile files in backend/
ls -la onnxruntime_profile_*.json
```

## Troubleshooting

### QNN Provider Not Available
```bash
# Check ONNX Runtime installation
python -c "import onnxruntime; print(onnxruntime.__version__)"
# Should be 1.16.0+ for QNN support

# Reinstall with QNN support
pip uninstall onnxruntime
pip install onnxruntime-qnn
```

### ConvInteger Errors
```bash
# Convert models to QOperator format
python convert_whisper_for_npu.py

# Verify conversion
python -c "
import onnx
model = onnx.load('whisper-small-onnx/onnx/encoder_model_npu.onnx')
ops = [op.op_type for op in model.graph.node]
print('ConvInteger in model:', 'ConvInteger' in ops)
"
```

### Performance Issues
1. **Check model format**: Ensure using NPU-optimized models
2. **Monitor memory**: NPU models may use more memory
3. **Batch size**: Adjust for optimal NPU utilization
4. **Model size**: Consider smaller models for faster inference

## Performance Expectations

### ASR Performance (Whisper Small)
- **CPU only**: ~2-3 seconds for 5-second audio
- **NPU accelerated**: ~0.5-1 second for 5-second audio
- **Accuracy**: Same as CPU version

### LLM Performance (Flan-T5 Small)
- **CPU only**: ~3-5 seconds per response
- **NPU accelerated**: ~1-2 seconds per response
- **Memory usage**: ~2-3GB for NPU models

## Advanced Configuration

### Custom QNN Settings
```python
# In asr.py, modify qnn_provider_options
qnn_provider_options = {
    "backend_path": "/path/to/qnn/backend",
    "device_id": 0,
    "htp_performance_mode": "burst",  # or "sustained"
    "htp_arch": "v73"  # or "v68"
}
```

### Model Optimization
```bash
# Further optimize models for specific use cases
python -c "
from onnxruntime.quantization import optimize_model
optimize_model('encoder_model_npu.onnx', 'encoder_model_optimized.onnx')
"
```

## Support

For NPU-specific issues:
1. Check Qualcomm QNN documentation
2. Verify ONNX Runtime QNN compatibility
3. Test with simpler models first
4. Monitor system resources

## Next Steps

1. **Production Deployment**: Set up proper service management
2. **Performance Tuning**: Optimize for your specific use case
3. **Model Updates**: Keep models updated for latest NPU features
4. **Monitoring**: Implement proper logging and monitoring 