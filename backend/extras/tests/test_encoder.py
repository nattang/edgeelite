import numpy as np
import onnxruntime as ort
import os
import time
import threading

# Test encoder inference with QNN provider
BASE_DIR = os.path.dirname(__file__)
ENCODER_PATH = os.path.join(BASE_DIR, "models/whisper_large_v3_turbo-hfwhisperencoder-qualcomm_snapdragon_x_elite.onnx/model.onnx/model.onnx")

print("=== QNN Encoder Test ===")
print(f"Encoder path: {ENCODER_PATH}")
print(f"File exists: {os.path.exists(ENCODER_PATH)}")

# Test 1: Load model with QNN
print("\n1. Loading encoder model with QNN...")
try:
    encoder_sess = ort.InferenceSession(ENCODER_PATH, providers=["QNNExecutionProvider"])
    print("✅ Encoder model loaded successfully with QNN")
    print(f"Available providers: {encoder_sess.get_providers()}")
    print(f"Active provider: {encoder_sess.get_providers()[0]}")
except Exception as e:
    print(f"❌ Failed to load encoder with QNN: {e}")
    exit(1)

# Test 2: Check model inputs
print("\n2. Checking model inputs...")
input_meta = encoder_sess.get_inputs()[0]
print(f"Input name: {input_meta.name}")
print(f"Input shape: {input_meta.shape}")
print(f"Input type: {input_meta.type}")

# Test 3: Run inference with exact expected shape
print("\n3. Testing with exact expected input shape (1, 128, 3000) and float16...")
input_shape = (1, 128, 3000)
dummy_input = np.random.randn(*input_shape).astype(np.float16)
print(f"  Input created: {dummy_input.shape}, dtype: {dummy_input.dtype}")

# Function to run inference
result_holder = {}
def run_inference():
    try:
        start_time = time.time()
        print(f"  Starting inference...")
        encoder_out = encoder_sess.run(None, {"input_features": dummy_input})[0]
        end_time = time.time()
        print(f"  ✅ Success! Output shape: {encoder_out.shape}")
        print(f"  ⏱️  Time taken: {end_time - start_time:.2f} seconds")
        result_holder['success'] = True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        result_holder['success'] = False

# Run inference in a thread with timeout
thread = threading.Thread(target=run_inference)
thread.start()
thread.join(timeout=30)
if thread.is_alive():
    print("  ❌ Inference is hanging (more than 30 seconds). Possible QNN or model issue.")
    thread.join()

print("\n=== Test Complete ===") 