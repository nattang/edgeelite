# Flan-T5 Model Setup for EdgeElite

This guide explains how to set up the Flan-T5-small model for local/offline use in the EdgeElite pipeline.

## Quick Start (Automated)

1. **Run the setup script:**
   ```bash
   cd backend
   python setup_flan_t5_models.py
   ```

2. **Follow the ONNX conversion instructions** provided by the script

3. **Test the setup:**
   ```bash
   python flan-t5-small-ONNX/test_model.py
   ```

4. **Run EdgeElite:**
   ```bash
   python main.py
   ```

## Manual Setup (Alternative)

If the automated script doesn't work, follow these manual steps:

### Step 1: Create Directory Structure
```bash
mkdir -p flan-t5-small-ONNX/onnx
```

### Step 2: Download Model Files
Download these files from [google/flan-t5-small](https://huggingface.co/google/flan-t5-small):
- `config.json`
- `generation_config.json`
- `special_tokens_map.json`
- `tokenizer_config.json`
- `tokenizer.json`

Place them in the `flan-t5-small-ONNX/` directory.

### Step 3: Convert to ONNX
Choose one of these methods:

#### Option A: Using Optimum (Recommended)
```bash
pip install optimum[onnxruntime]
python -m optimum.exporters.onnx --model google/flan-t5-small --task text2text-generation --framework pt2 --output flan-t5-small-ONNX/onnx
```

#### Option B: Using Transformers
```bash
pip install transformers[onnx]
python -c "
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.onnx import export
import torch

model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

export(tokenizer, model, 'flan-t5-small-ONNX/onnx', opset=12)
"
```

### Step 4: Verify Setup
```bash
python flan-t5-small-ONNX/test_model.py
```

## Directory Structure

After setup, your directory should look like:
```
backend/
├── flan-t5-small-ONNX/
│   ├── config.json
│   ├── generation_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── test_model.py
│   └── onnx/
│       ├── encoder_model.onnx
│       ├── decoder_model.onnx
│       └── ... (other ONNX files)
└── main.py
```

## Troubleshooting

### Common Issues

1. **"Local tokenizer loading failed"**
   - This is normal! The tokenizer will fall back to HuggingFace hub
   - Your files are downloaded correctly

2. **ONNX conversion errors**
   - Ensure you have PyTorch installed: `pip install torch`
   - Try different ONNX opset versions (11, 12, 13)
   - Check disk space (need ~100MB)

3. **Model loading errors**
   - Verify all required files are present
   - Check file permissions
   - Ensure Python environment has all dependencies

4. **NPU/QNN acceleration issues**
   - Use quantized ONNX models for NPU
   - Install onnxruntime-qnn for Qualcomm NPU support
   - Check QNN provider availability

### Performance Tips

- **CPU**: Use FP32 ONNX models for best compatibility
- **NPU**: Use INT8 quantized models for acceleration
- **Memory**: Flan-T5-small uses ~80M parameters (~100MB RAM)

## Model Information

- **Model**: google/flan-t5-small
- **Parameters**: 80M
- **Task**: Text-to-text generation
- **License**: Apache 2.0
- **Size**: ~100MB (including ONNX files)

## Alternative Models

You can replace Flan-T5 with other models:
- **T5-small**: Similar architecture, different training
- **Flan-T5-base**: Larger model (250M parameters)
- **Custom models**: Convert your own models to ONNX

## Support

- **EdgeElite Issues**: Check project repository
- **Model Issues**: [HuggingFace Flan-T5](https://huggingface.co/google/flan-t5-small)
- **ONNX Issues**: [ONNX Runtime Documentation](https://onnxruntime.ai/)

## License

This setup script and documentation are part of the EdgeElite project.
The Flan-T5 model is licensed under Apache 2.0. 