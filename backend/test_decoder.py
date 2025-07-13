import numpy as np
import onnxruntime as ort
import os
import time
import threading

# Test decoder inference with QNN provider
BASE_DIR = os.path.dirname(__file__)
DECODER_PATH = os.path.join(BASE_DIR, "models/whisper_large_v3_turbo-hfwhisperdecoder-qualcomm_snapdragon_x_elite.onnx/model.onnx/model.onnx")

print("=== QNN Decoder Test ===")
print(f"Decoder path: {DECODER_PATH}")
print(f"File exists: {os.path.exists(DECODER_PATH)}")

# Test 1: Load model with QNN
print("\n1. Loading decoder model with QNN...")
try:
    decoder_sess = ort.InferenceSession(DECODER_PATH, providers=["QNNExecutionProvider"])
    print("✅ Decoder model loaded successfully with QNN")
    print(f"Available providers: {decoder_sess.get_providers()}")
    print(f"Active provider: {decoder_sess.get_providers()[0]}")
except Exception as e:
    print(f"❌ Failed to load decoder with QNN: {e}")
    exit(1)

# Test 2: Check model inputs
print("\n2. Checking model inputs...")
inputs = decoder_sess.get_inputs()
for i, input_meta in enumerate(inputs):
    print(f"Input {i}: name={input_meta.name}, shape={input_meta.shape}, type={input_meta.type}")

# Test 3: Create dummy encoder output (from encoder test)
print("\n3. Creating dummy encoder output...")
# From encoder test: output shape was (20, 1, 64, 1500)
encoder_out = np.random.randn(20, 1, 64, 1500).astype(np.float16)
print(f"Encoder output shape: {encoder_out.shape}, dtype: {encoder_out.dtype}")

# Test 4: Prepare decoder inputs
print("\n4. Preparing decoder inputs...")
input_ids = np.array([[50257]], dtype=np.int32)  # Start token only, shape [1, 1]
position_ids = np.array([0], dtype=np.int32)  # Position ID

# Create attention mask
attention_mask = np.ones((1, 1, 1, 200), dtype=np.float16)

# Create cache inputs for 4 layers
k_cache_self = []
v_cache_self = []
k_cache_cross = []
v_cache_cross = []

for i in range(4):
    # Self-attention cache (starts empty)
    k_cache_self.append(np.zeros((20, 1, 64, 199), dtype=np.float16))
    v_cache_self.append(np.zeros((20, 1, 199, 64), dtype=np.float16))
    
    # Cross-attention cache (from encoder)
    k_cache_cross.append(np.random.randn(20, 1, 64, 1500).astype(np.float16))
    v_cache_cross.append(np.random.randn(20, 1, 1500, 64).astype(np.float16))

# Prepare inputs for decoder
inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "k_cache_self_0_in": k_cache_self[0],
    "v_cache_self_0_in": v_cache_self[0],
    "k_cache_self_1_in": k_cache_self[1],
    "v_cache_self_1_in": v_cache_self[1],
    "k_cache_self_2_in": k_cache_self[2],
    "v_cache_self_2_in": v_cache_self[2],
    "k_cache_self_3_in": k_cache_self[3],
    "v_cache_self_3_in": v_cache_self[3],
    "k_cache_cross_0": k_cache_cross[0],
    "v_cache_cross_0": v_cache_cross[0],
    "k_cache_cross_1": k_cache_cross[1],
    "v_cache_cross_1": v_cache_cross[1],
    "k_cache_cross_2": k_cache_cross[2],
    "v_cache_cross_2": v_cache_cross[2],
    "k_cache_cross_3": k_cache_cross[3],
    "v_cache_cross_3": v_cache_cross[3]
}

print(f"Input shapes:")
for name, value in inputs.items():
    print(f"  {name}: {value.shape}, dtype: {value.dtype}")

# Test 5: Run decoder inference
print("\n5. Testing decoder inference...")

# Function to run inference
result_holder = {}
def run_inference():
    try:
        start_time = time.time()
        print(f"  Starting decoder inference...")
        outputs = decoder_sess.run(None, inputs)
        end_time = time.time()
        print(f"  ✅ Success! Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"    Output {i} shape: {output.shape}, dtype: {output.dtype}")
        print(f"  ⏱️  Time taken: {end_time - start_time:.2f} seconds")
        result_holder['success'] = True
        result_holder['outputs'] = outputs
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        result_holder['success'] = False

# Run inference in a thread with timeout
thread = threading.Thread(target=run_inference)
thread.start()
thread.join(timeout=60)  # 60 seconds timeout for decoder
if thread.is_alive():
    print("  ❌ Decoder inference is hanging (more than 60 seconds).")
    thread.join()

print("\n=== Test Complete ===") 