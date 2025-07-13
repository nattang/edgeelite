import onnxruntime as ort
from PIL import Image
import numpy as np

class TrocrApp:
    def __init__(self, encoder_session: ort.InferenceSession, decoder_session: ort.InferenceSession):
        self.encoder_session = encoder_session
        self.decoder_session = decoder_session

    def process_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB").resize((384, 384))
        img = np.array(image).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize as expected by TrOCR
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return img

    def encoder_preprocess(self, processed_image: np.ndarray):
        encoder_outputs = self.encoder_session.run(None, {"pixel_values": processed_image})
        return encoder_outputs

    def create_empty_kv_cache(self, num_layers=6, heads=8, dim=32, seq_len=19):
        kv = []
        for _ in range(num_layers):
            key = np.zeros((1, heads, seq_len, dim), dtype=np.float32)
            val = np.zeros((1, heads, seq_len, dim), dtype=np.float32)
            kv.extend([key, val])
        return kv

    def encoder_to_decoder_cache(self, encoder_outputs, decoder_start_token: int, num_layers=6, num_heads=8, kv_dim: int=32):
        input_ids = np.array([[decoder_start_token]], dtype=np.int32)
        index = np.array([0], dtype=np.int32)

        inputs = {
            "input_ids": input_ids,
            "index": index
        }

        # Ensure encoder_outputs are NumPy arrays (not PyTorch tensors)
        encoder_outputs = [out if isinstance(out, np.ndarray) else out.cpu().numpy() for out in encoder_outputs]

        # Add cross-attention from encoder output
        for layer in range(num_layers):
            inputs[f"kv_{layer}_cross_attn_key"] = encoder_outputs[2 * layer]
            inputs[f"kv_{layer}_cross_attn_val"] = encoder_outputs[2 * layer + 1]

        # Add self-attention as empty
        empty_kv = self.create_empty_kv_cache(num_layers, num_heads, kv_dim, seq_len=19)
        for layer in range(num_layers):
            inputs[f"kv_{layer}_attn_key"] = empty_kv[2 * layer]
            inputs[f"kv_{layer}_attn_val"] = empty_kv[2 * layer + 1]

        self.check_decoder_input_shapes(inputs, self.decoder_session)

        return inputs   

    def check_decoder_input_shapes(self, decoder_inputs, decoder_session):
        print("\n--- Decoder Input Debug ---")
        expected_inputs = {inp.name: inp for inp in decoder_session.get_inputs()}

        for name, value in decoder_inputs.items():
            print(f"Input name: {name}")
            print(f"  Provided shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
            print(f"  Provided dtype: {value.dtype if hasattr(value, 'dtype') else type(value)}")

            if name in expected_inputs:
                expected = expected_inputs[name]
                print(f"  Expected shape: {expected.shape}")
                print(f"  Expected dtype: {expected.type}")
            else:
                print("  ⚠️ Warning: This input is not found in decoder_session.get_inputs()!")

            print()

        # Check for missing required inputs
        missing_inputs = set(expected_inputs.keys()) - set(decoder_inputs.keys())
        if missing_inputs:
            print(f"❌ Missing inputs: {missing_inputs}")
        else:
            print("✅ All required inputs are present.")


    def decoder_preprocess(self, image_path: str):
        pass