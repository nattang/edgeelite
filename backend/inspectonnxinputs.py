import onnxruntime as ort

encoder_path = "models/asr/encoder_model.onnx"
decoder_path = "models/asr/decoder_model_merged.onnx"

# Load ONNX sessions
encoder_session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
decoder_session = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])

# Print encoder input names
print("🔍 Encoder Input Names:")
for inp in encoder_session.get_inputs():
    print(f"- {inp.name}: {inp.shape} | {inp.type}")

# Print decoder input names
print("\n🔍 Decoder Input Names:")
for inp in decoder_session.get_inputs():
    print(f"- {inp.name}: {inp.shape} | {inp.type}")
