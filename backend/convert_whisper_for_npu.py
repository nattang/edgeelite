#!/usr/bin/env python3
"""
Convert Whisper ONNX models to QOperator format for Snapdragon X-Elite NPU compatibility
This script converts ConvInteger ops to QLinearConv ops that QNN EP supports.
"""

import os
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from pathlib import Path

class WhisperCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for Whisper models"""
    def __init__(self, model_path, num_samples=10):
        self.model_path = model_path
        self.num_samples = num_samples
        self.current_sample = 0
        
        # Load model to get input info
        self.model = onnx.load(model_path)
        self.input_name = self.model.graph.input[0].name
        self.input_shape = None
        for input_info in self.model.graph.input:
            if input_info.name == self.input_name:
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        if self.input_shape is None:
                            self.input_shape = [dim.dim_value]
                        else:
                            self.input_shape.append(dim.dim_value)
                break
    
    def get_next(self):
        if self.current_sample >= self.num_samples:
            return None
            
        # Generate dummy calibration data
        if "encoder" in self.model_path:
            # Encoder expects mel spectrogram: (batch, 80, 3000)
            data = np.random.randn(1, 80, 3000).astype(np.float32)
        else:
            # Decoder expects tokens: (batch, seq_len) - vary sequence length
            seq_len = np.random.randint(1, 10)
            data = np.random.randint(0, 1000, (1, seq_len)).astype(np.int64)
        
        self.current_sample += 1
        return {self.input_name: data}

def convert_model_for_npu(input_path, output_path, model_type="encoder"):
    """Convert a Whisper model to QOperator format for NPU compatibility"""
    print(f"🔄 Converting {model_type} model for NPU...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    
    # Create calibration data reader
    calibration_reader = WhisperCalibrationDataReader(input_path, num_samples=5)
    
    # Quantize to QOperator format
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QOperator,  # This produces QLinearConv instead of ConvInteger
        weight_type=QuantType.QInt8,
        per_channel=True,
        optimize_model=True
    )
    
    print(f"✅ {model_type} model converted successfully!")
    
    # Verify the converted model
    converted_model = onnx.load(output_path)
    conv_integer_ops = [op for op in converted_model.graph.node if op.op_type == "ConvInteger"]
    qlinear_conv_ops = [op for op in converted_model.graph.node if op.op_type == "QLinearConv"]
    
    print(f"   ConvInteger ops: {len(conv_integer_ops)}")
    print(f"   QLinearConv ops: {len(qlinear_conv_ops)}")
    
    if len(conv_integer_ops) == 0 and len(qlinear_conv_ops) > 0:
        print("   ✅ Model is NPU-compatible!")
    else:
        print("   ⚠️ Model may still have compatibility issues")

def main():
    """Main conversion function"""
    print("🚀 Whisper NPU Model Converter")
    print("=" * 50)
    
    # Model paths
    model_dir = Path(__file__).parent / "whisper-small-onnx" / "onnx"
    
    # Input models (original INT8 with ConvInteger)
    encoder_input = model_dir / "encoder_model_int8.onnx"
    decoder_input = model_dir / "decoder_model_int8.onnx"
    
    # Output models (QOperator format for NPU)
    encoder_output = model_dir / "encoder_model_npu.onnx"
    decoder_output = model_dir / "decoder_model_npu.onnx"
    
    # Check if input models exist
    if not encoder_input.exists():
        print(f"❌ Encoder model not found: {encoder_input}")
        return
    
    if not decoder_input.exists():
        print(f"❌ Decoder model not found: {decoder_input}")
        return
    
    print("📁 Input models found:")
    print(f"   Encoder: {encoder_input}")
    print(f"   Decoder: {decoder_input}")
    print()
    
    # Convert encoder
    if encoder_input.exists():
        convert_model_for_npu(encoder_input, encoder_output, "encoder")
        print()
    
    # Convert decoder
    if decoder_input.exists():
        convert_model_for_npu(decoder_input, decoder_output, "decoder")
        print()
    
    print("🎯 NPU Conversion Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Copy the converted models to your Snapdragon X-Elite device")
    print("2. Install onnxruntime-qnn on the device")
    print("3. Update ASR service to use the NPU models")
    print("4. Test with QNNExecutionProvider")

if __name__ == "__main__":
    main() 