#!/usr/bin/env python3
"""
ASR Service for EdgeElite Backend - QNN/CPU Fallback, Qualcomm-style
Optimized for Snapdragon X-Elite NPU, with robust fallback and detailed logs.
"""

import os
import numpy as np
import librosa
import onnxruntime as ort
from pathlib import Path
from transformers import WhisperTokenizer

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "whisper-small-onnx", "onnx")
# NPU-optimized models (QOperator format)
ENCODER_PATH_NPU = os.path.join(MODEL_DIR, "encoder_model_npu.onnx")
DECODER_PATH_NPU = os.path.join(MODEL_DIR, "decoder_model_npu.onnx")
# Fallback models
ENCODER_PATH_INT8 = os.path.join(MODEL_DIR, "encoder_model_int8.onnx")
DECODER_PATH_INT8 = os.path.join(MODEL_DIR, "decoder_model_int8.onnx")
ENCODER_PATH_FP32 = os.path.join(MODEL_DIR, "encoder_model.onnx")
DECODER_PATH_FP32 = os.path.join(MODEL_DIR, "decoder_model.onnx")

# Check which models exist
def get_available_models():
    """Check which models are available and return paths"""
    models = {
        'int8': {'encoder': None, 'decoder': None},
        'fp32': {'encoder': None, 'decoder': None}
    }
    
    if os.path.exists(ENCODER_PATH_INT8):
        models['int8']['encoder'] = ENCODER_PATH_INT8
    if os.path.exists(DECODER_PATH_INT8):
        models['int8']['decoder'] = DECODER_PATH_INT8
    if os.path.exists(ENCODER_PATH_FP32):
        models['fp32']['encoder'] = ENCODER_PATH_FP32
    if os.path.exists(DECODER_PATH_FP32):
        models['fp32']['decoder'] = DECODER_PATH_FP32
    
    return models

# Tokenizer - use regular Whisper tokenizer for text output
TOKENIZER = WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")

# QNN provider options (Qualcomm style)
onnxruntime_dir = Path(ort.__file__).parent
hexagon_driver = onnxruntime_dir / "capi" / "QnnHtp.dll"
qnn_provider_options = {"backend_path": hexagon_driver}

# Session options
so = ort.SessionOptions()
so.enable_profiling = False  # Disable profiling to reduce file clutter
so.log_severity_level = 3

# Encoder/Decoder sessions (lazy init)
_sess_enc = None
_sess_dec = None
_encoder_provider = None
_decoder_provider = None

# Provider lists
QNN_PROVIDERS = [("QNNExecutionProvider", qnn_provider_options), "CPUExecutionProvider"]
CPU_PROVIDERS = ["CPUExecutionProvider"]

def _init_sessions():
    global _sess_enc, _sess_dec, _encoder_provider, _decoder_provider
    # Encoder - try NPU models first, then fallback
    if _sess_enc is None:
        encoder_paths = [
            (ENCODER_PATH_NPU, "NPU-optimized"),
            (ENCODER_PATH_INT8, "INT8"),
            (ENCODER_PATH_FP32, "FP32")
        ]
        
        for path, model_type in encoder_paths:
            if os.path.exists(path):
                try:
                    print(f"🚀 Trying to load {model_type} encoder...")
                    if model_type == "NPU-optimized":
                        _sess_enc = ort.InferenceSession(path, providers=QNN_PROVIDERS, sess_options=so)
                    else:
                        _sess_enc = ort.InferenceSession(path, providers=CPU_PROVIDERS, sess_options=so)
                    _encoder_provider = _sess_enc.get_providers()[0]
                    print(f"✅ Encoder loaded with provider: {_encoder_provider} ({model_type})")
                    break
                except Exception as e:
                    print(f"⚠️ {model_type} encoder failed: {e}")
                    continue
        else:
            raise RuntimeError("No encoder model could be loaded")
    
    # Decoder - try NPU models first, then fallback
    if _sess_dec is None:
        decoder_paths = [
            (DECODER_PATH_NPU, "NPU-optimized"),
            (DECODER_PATH_INT8, "INT8"),
            (DECODER_PATH_FP32, "FP32")
        ]
        
        for path, model_type in decoder_paths:
            if os.path.exists(path):
                try:
                    print(f"🚀 Trying to load {model_type} decoder...")
                    if model_type == "NPU-optimized":
                        _sess_dec = ort.InferenceSession(path, providers=QNN_PROVIDERS, sess_options=so)
                    else:
                        _sess_dec = ort.InferenceSession(path, providers=CPU_PROVIDERS, sess_options=so)
                    _decoder_provider = _sess_dec.get_providers()[0]
                    print(f"✅ Decoder loaded with provider: {_decoder_provider} ({model_type})")
                    break
                except Exception as e:
                    print(f"⚠️ {model_type} decoder failed: {e}")
                    continue
        else:
            raise RuntimeError("No decoder model could be loaded")

def transcribe_audio(filename):
    print("[LOG] Starting ASR transcription")
    _init_sessions()
    print(f"[DEMO] Encoder provider: {_encoder_provider}, Decoder provider: {_decoder_provider}")
    
    # Load and preprocess audio
    print(f"[LOG] Loading audio: {filename}")
    wav, sr = librosa.load(filename, sr=16000)
    print(f"[LOG] Audio loaded, length: {len(wav)/sr:.2f}s, sample rate: {sr}")
    
    # Check audio quality and silence
    audio_energy = np.mean(np.abs(wav))
    print(f"[LOG] Audio energy: {audio_energy:.4f}")
    
    if audio_energy < 0.01:
        print("⚠️ Audio appears to be mostly silence")
        return [{"start": 0, "end": float(len(wav))/sr, "text": "[No speech detected - audio too quiet]"}]
    
    # Normalize audio
    wav = wav / (np.max(np.abs(wav)) + 1e-8)
    
    # Apply simple noise reduction (high-pass filter to remove low-frequency noise)
    from scipy import signal
    # High-pass filter to remove low-frequency noise
    b, a = signal.butter(4, 100, btype='high', fs=sr)
    wav = signal.filtfilt(b, a, wav)
    
    # Apply simple gain if audio is too quiet
    if audio_energy < 0.05:
        wav = wav * 2.0  # Amplify quiet audio
        print("[LOG] Amplified quiet audio")
    
    # Limit audio length for faster processing (max 30 seconds)
    max_length = 30 * sr  # 30 seconds
    if len(wav) > max_length:
        wav = wav[:max_length]
        print(f"⚠️ Audio truncated to {max_length/sr:.1f}s for faster processing")
    
    # Extract mel spectrogram
    print("[LOG] Extracting mel spectrogram...")
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=80, hop_length=160, n_fft=400
    )
    log_mel = np.log1p(mel).astype(np.float32)
    print(f"[LOG] Raw mel spectrogram shape: {log_mel.shape}")
    
    # Whisper expects 3000 time steps - pad or truncate
    expected_time_steps = 3000
    current_time_steps = log_mel.shape[1]
    
    if current_time_steps < expected_time_steps:
        # Pad with zeros
        padding = np.zeros((80, expected_time_steps - current_time_steps), dtype=np.float32)
        log_mel = np.concatenate([log_mel, padding], axis=1)
        print(f"[LOG] Padded mel spectrogram from {current_time_steps} to {expected_time_steps} time steps")
    elif current_time_steps > expected_time_steps:
        # Truncate
        log_mel = log_mel[:, :expected_time_steps]
        print(f"[LOG] Truncated mel spectrogram from {current_time_steps} to {expected_time_steps} time steps")
    
    # Add batch dimension
    log_mel = log_mel[None, :, :]
    print(f"[LOG] Final mel spectrogram shape: {log_mel.shape}")
    
    # Run encoder
    print("[LOG] Running encoder...")
    encoder_output = _sess_enc.run(None, {_sess_enc.get_inputs()[0].name: log_mel})[0]
    print(f"[LOG] Encoder output shape: {encoder_output.shape}")
    
    # Fast greedy decoding with early stopping
    # Start with <|startoftranscript|> token and force transcription task
    decoded = [50256]  # <|startoftranscript|>
    
    # Force transcription task (not translation) and regular text (not phonetic)
    # Whisper task tokens: 50258=<|transcribe|>, 50358=<|translate|>
    # We want transcription, not translation, and regular text, not phonetic
    decoded.append(50258)  # <|transcribe|>
    
    # Add language token for English (optional, but helps with accuracy)
    decoded.append(50363)  # <|en|> (English)
    
    # Get decoder input names
    decoder_input_names = [input.name for input in _sess_dec.get_inputs()]
    print(f"[LOG] Decoder input names: {decoder_input_names}")
    
    # Limit decoding steps for speed
    max_steps = 30  # Reduced for faster processing
    print(f"[LOG] Starting decoder loop (max_steps={max_steps})...")
    
    for step in range(max_steps):
        print(f"[LOG] Decoder step {step}, current tokens: {decoded}")
        
        # Prepare input tokens for this step
        token = np.array([decoded], dtype=np.int64)
        
        # Run decoder - fix input mapping
        input_ids_name = _sess_dec.get_inputs()[0].name  # 'input_ids' for tokens
        encoder_hidden_states_name = _sess_dec.get_inputs()[1].name  # 'encoder_hidden_states' for encoder output
        
        print(f"[LOG] Decoder inputs - input_ids: {input_ids_name}, encoder_hidden_states: {encoder_hidden_states_name}")
        print(f"[LOG] Encoder output dtype: {encoder_output.dtype}, token dtype: {token.dtype}")
        
        # Pass inputs in correct order
        logits = _sess_dec.run(
            None, {
                input_ids_name: token,  # tokens go to input_ids
                encoder_hidden_states_name: encoder_output  # encoder output goes to encoder_hidden_states
            }
        )[0]
        print(f"[LOG] Decoder logits shape: {logits.shape}")
        
        # Get next token from the last position
        next_token = np.argmax(logits[0, -1, :])
        print(f"[LOG] Next token: {next_token}")
        
        # Stop if end token, but only after some content
        if next_token == 50257:  # <|endoftranscript|>
            if len(decoded) <= 4:  # Only special tokens, no actual content
                print("[LOG] End token reached too early, continuing to look for speech...")
                # Try to continue decoding by masking the end token
                logits_copy = logits[0, -1, :].copy()
                logits_copy[50257] = -float('inf')  # Mask end token
                next_token = np.argmax(logits_copy)
                print(f"[LOG] Continuing with token: {next_token}")
            else:
                print("[LOG] End of transcript token reached, stopping decoding loop.")
                break
        
        # Filter out phonetic tokens (common IPA tokens that produce phonetic output)
        phonetic_token_ids = [220, 135, 230, 74, 134, 108]  # Common IPA tokens
        if next_token in phonetic_token_ids:
            print(f"[LOG] Skipping phonetic token: {next_token}")
            # Try to get the next best token
            logits_copy = logits[0, -1, :].copy()
            logits_copy[phonetic_token_ids] = -float('inf')  # Mask out phonetic tokens
            next_token = np.argmax(logits_copy)
            print(f"[LOG] Using alternative token: {next_token}")
            
        # Add to decoded sequence
        decoded.append(int(next_token))
        
        # Early stopping if stuck on same token
        if step > 3 and len(set(decoded[-3:])) == 1:
            print(f"[LOG] Early stopping: stuck on token {decoded[-1]}")
            break
        
        # Early stopping if no meaningful progress
        if step > 15 and len(decoded) < 6:
            print("[LOG] Early stopping: no meaningful progress in decoding loop.")
            break
        
        # Force minimum content before allowing end token
        if step < 8 and next_token == 50257:  # Increased from 5 to 8
            print("[LOG] Forcing more content before allowing end token...")
            logits_copy = logits[0, -1, :].copy()
            logits_copy[50257] = -float('inf')  # Mask end token
            next_token = np.argmax(logits_copy)
            print(f"[LOG] Forced token: {next_token}")
        
        # Also force more content if we have very few tokens
        if len(decoded) < 8 and next_token == 50257:
            print("[LOG] Forcing more content due to short transcript...")
            logits_copy = logits[0, -1, :].copy()
            logits_copy[50257] = -float('inf')  # Mask end token
            next_token = np.argmax(logits_copy)
            print(f"[LOG] Forced token: {next_token}")
        
        # Early stopping if we're getting too many phonetic tokens
        phonetic_tokens = [t for t in decoded if t in [220, 135, 230, 74, 134, 108]]  # Common IPA tokens
        if len(phonetic_tokens) > 10:
            print("[LOG] Early stopping: too many phonetic tokens detected.")
            break
    
    # Decode to text - skip special tokens
    print(f"[LOG] Decoding tokens to text: {decoded}")
    
    # Filter out special tokens for actual text
    text_tokens = [t for t in decoded if t not in [50256, 50257, 50258, 50358, 50359, 50360, 50361, 50362]]
    
    if text_tokens:
        text = TOKENIZER.decode(text_tokens)
        text = text.strip()
        
        # Clean up common artifacts
        text = text.replace("(Music", "").replace("(music", "")
        text = text.replace("(Applause", "").replace("(applause", "")
        text = text.replace("(Laughter", "").replace("(laughter", "")
        text = text.replace("(Background", "").replace("(background", "")
        
        # Remove leading/trailing punctuation
        text = text.strip(" .,!?;:")
        
        # If text is too short, it might be noise
        if len(text) < 3:
            text = "[No clear speech detected]"
    else:
        text = "[No speech detected]"
    
    print(f"🎤 Transcribed: '{text}'")
    print("[LOG] ASR transcription complete.")
    return [{"start": 0, "end": float(len(wav))/sr, "text": text}]
