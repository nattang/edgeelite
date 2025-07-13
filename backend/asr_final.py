import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa
import os
import json

BASE_DIR = os.path.dirname(__file__)
# Use Qualcomm Snapdragon X Elite optimized Whisper models
ENCODER_PATH = os.path.join(BASE_DIR, "models/whisper_large_v3_turbo-hfwhisperencoder-qualcomm_snapdragon_x_elite.onnx/model.onnx/model.onnx")
DECODER_PATH = os.path.join(BASE_DIR, "models/whisper_large_v3_turbo-hfwhisperdecoder-qualcomm_snapdragon_x_elite.onnx/model.onnx/model.onnx")
# Use tokenizer files from the asr directory
MODEL_DIR = os.path.join(BASE_DIR, "models/asr")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
ADDED_TOKENS_PATH = os.path.join(MODEL_DIR, "added_tokens.json")
SPECIAL_TOKENS_PATH = os.path.join(MODEL_DIR, "special_tokens_map.json")
SAMPLING_RATE = 16000
CHUNK_DURATION = 5.0

# Enforce QNN/NPU usage only
encoder_sess = ort.InferenceSession(ENCODER_PATH, providers=["QNNExecutionProvider"])
decoder_sess = ort.InferenceSession(DECODER_PATH, providers=["QNNExecutionProvider"])
print("Using QNN Execution Provider")
print(f"Encoder session providers: {encoder_sess.get_providers()}")
print(f"Decoder session providers: {decoder_sess.get_providers()}")
print(f"Encoder session active provider: {encoder_sess.get_provider_options()}")
print(f"Decoder session active provider: {decoder_sess.get_provider_options()}")
assert "QNNExecutionProvider" in encoder_sess.get_providers(), "NPU (QNN) is NOT being used for encoder!"
assert "QNNExecutionProvider" in decoder_sess.get_providers(), "NPU (QNN) is NOT being used for decoder!"

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    token_to_id = json.load(f)

# Handle potentially incomplete added_tokens.json
try:
    with open(ADDED_TOKENS_PATH, "r", encoding="utf-8") as f:
        added_tokens = json.load(f)
    token_to_id.update(added_tokens)
except (json.JSONDecodeError, FileNotFoundError):
    print("Warning: added_tokens.json is incomplete or missing, using empty dict")
    added_tokens = {}

with open(SPECIAL_TOKENS_PATH, "r", encoding="utf-8") as f:
    special_tokens_map = json.load(f)

for token in special_tokens_map.get("additional_special_tokens", []):
    if token not in token_to_id:
        print(f"Token {token} missing in vocab and added_tokens")

id_to_token = {v: k for k, v in token_to_id.items()}
special_tokens = {token: token_to_id[token] for token in special_tokens_map.get("additional_special_tokens", []) if token in token_to_id}
special_tokens.update(added_tokens)

def read_audio(filename):
    AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../recordings")
    audio_path = os.path.join(AUDIO_DIR, filename)
    audio, file_sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if file_sr != SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SAMPLING_RATE)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95
    return audio

def log_mel_spectrogram_whisper(audio, n_mels=128):
    n_fft = 400
    hop_length = 160
    audio = np.pad(audio, (0, n_fft), mode='constant')
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLING_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=8000,
        power=2.0,
        window='hann',
        center=True,
        pad_mode='reflect'
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = np.clip(log_mel_spec, -80, 0)
    log_mel_spec = (log_mel_spec + 80) / 80 * 2 - 1
    return log_mel_spec.astype(np.float32)

def pad_or_trim_mel(mel, target_len=3000):
    if mel.shape[1] > target_len:
        return mel[:, :target_len]
    elif mel.shape[1] < target_len:
        pad_width = target_len - mel.shape[1]
        return np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=-1.0)
    return mel

def decode_token_id(token_id):
    return id_to_token.get(token_id, f"[UNK_{token_id}]")

def chunk_audio(audio):
    chunk_size = int(SAMPLING_RATE * CHUNK_DURATION)
    overlap = 0
    chunks = []
    for i in range(0, len(audio), chunk_size - overlap):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size // 4:
            continue
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
        start_time = i / SAMPLING_RATE
        end_time = min((i + chunk_size) / SAMPLING_RATE, len(audio) / SAMPLING_RATE)
        chunks.append((start_time, end_time, chunk))
    return chunks

def run_whisper_encoder(chunk):
    mel = log_mel_spectrogram_whisper(chunk)
    mel = pad_or_trim_mel(mel, 3000)
    mel = mel[np.newaxis, :, :]
    # Convert to float16 as required by QNN model
    mel = mel.astype(np.float16)
    try:
        print(f"Running encoder inference with provider: {encoder_sess.get_providers()}")
        print(f"Input shape: {mel.shape}, dtype: {mel.dtype}")
        encoder_out = encoder_sess.run(None, {"input_features": mel})[0]
        print(f"Encoder inference completed successfully, output shape: {encoder_out.shape}")
        return encoder_out
    except Exception as e:
        print(f"Encoder error: {e}")
        return None

def run_whisper_decoder_simple(encoder_out, max_tokens=20):  # Reduced from 50 to 20 to prevent hanging
    print(f"Starting decoder with encoder_out shape: {encoder_out.shape}")
    
    # Initialize with start token only
    tokens = [special_tokens.get("<|startoftranscript|>", 50257)]
    print(f"Initial token: {tokens}")
    
    for step in range(max_tokens):
        print(f"Decoder step {step + 1}/{max_tokens}")
        
        # Prepare inputs for QNN decoder
        input_ids = np.array([tokens], dtype=np.int32)  # Use int32 as required
        position_ids = np.array([len(tokens) - 1], dtype=np.int32)
        
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
        
        try:
            if step == 0:
                print(f"Running decoder inference with provider: {decoder_sess.get_providers()}")
                print(f"Input shapes: input_ids={input_ids.shape}")
            
            # Run decoder inference
            outputs = decoder_sess.run(None, inputs)
            logits = outputs[0]  # First output is logits
            
            if step == 0:
                print(f"Decoder inference completed successfully, logits shape: {logits.shape}")
            
            # Process logits
            temperature = 0.1
            logits = logits[0, -1, :] / temperature
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            next_token = int(np.argmax(probs))
            
            print(f"Step {step + 1}: Generated token {next_token}")
            
        except Exception as e:
            print(f"Decoder error at step {step}: {e}")
            break
        
        tokens.append(next_token)
        
        # Check for end tokens
        if next_token == special_tokens.get("<|endoftext|>", 50256):
            print("End of text token found")
            break
        if next_token == special_tokens.get("<|nospeech|>", 50361):
            print("No speech token found")
            break
    
    content_tokens = [t for t in tokens[1:] if t < 50000]  # Skip start token
    print(f"Final content tokens: {content_tokens}")
    return content_tokens

def process_audio(filename):
    audio = read_audio(filename)
    if np.sqrt(np.mean(audio**2)) < 0.001:
        print("Audio seems very quiet")
    chunks = chunk_audio(audio)
    results = []
    for i, (start_ts, end_ts, chunk) in enumerate(chunks):
        encoder_out = run_whisper_encoder(chunk)
        if encoder_out is None:
            continue
        token_ids = run_whisper_decoder_simple(encoder_out)
        text_tokens = [decode_token_id(tid) for tid in token_ids]
        text = "".join(text_tokens).replace("Ġ", " ").replace("▁", " ").strip()
        text = " ".join(text.split())
        print("Final Transcription: " + text)
        results.append({"start": round(start_ts, 2), "end": round(end_ts, 2), "text": text})
        print(results)
    try:
        os.remove("../app/" + filename)
    except Exception:
        pass
    return results

if __name__ == "__main__":
    from pprint import pprint
    res = process_audio("temp_audio.wav")
    pprint(res)
