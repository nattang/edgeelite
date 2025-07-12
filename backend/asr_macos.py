import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa
import os
import json

BASE_DIR = os.path.dirname(__file__)
ENCODER_PATH = os.path.join(BASE_DIR, "models/asr/encoder_model.onnx")
DECODER_PATH = os.path.join(BASE_DIR, "models/asr/decoder_model_merged.onnx")
VOCAB_PATH = os.path.join(BASE_DIR, "models/asr/vocab.json")
ADDED_TOKENS_PATH = os.path.join(BASE_DIR, "models/asr/added_tokens.json")
SPECIAL_TOKENS_PATH = os.path.join(BASE_DIR, "models/asr/special_tokens_map.json")
SAMPLING_RATE = 16000
CHUNK_DURATION = 5.0

encoder_sess = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
decoder_sess = ort.InferenceSession(DECODER_PATH, providers=["CPUExecutionProvider"])

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    token_to_id = json.load(f)
with open(ADDED_TOKENS_PATH, "r", encoding="utf-8") as f:
    added_tokens = json.load(f)
token_to_id.update(added_tokens)
with open(SPECIAL_TOKENS_PATH, "r", encoding="utf-8") as f:
    special_tokens_map = json.load(f)

for token in special_tokens_map.get("additional_special_tokens", []):
    if token not in token_to_id:
        print(f"Token {token} missing in vocab and added_tokens")

id_to_token = {v: k for k, v in token_to_id.items()}
special_tokens = {token: token_to_id[token] for token in special_tokens_map.get("additional_special_tokens", []) if token in token_to_id}
special_tokens.update(added_tokens)

def read_audio(filename):
    AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../app")
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
    overlap = int(chunk_size * 0.1)
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
    try:
        encoder_out = encoder_sess.run(None, {"input_features": mel})[0]
        return encoder_out
    except Exception as e:
        print(f"Encoder error: {e}")
        return None

def run_whisper_decoder_simple(encoder_out, max_tokens=50):
    tokens = [
        special_tokens.get("<|startoftranscript|>", 50257),
        special_tokens.get("<|en|>", 50259),
        special_tokens.get("<|transcribe|>", 50358),
        special_tokens.get("<|notimestamps|>", 50363)
    ]
    for step in range(max_tokens):
        input_ids = np.array([tokens], dtype=np.int64)
        batch_size = 1
        num_heads = 20
        head_dim = 64
        encoder_seq_len = encoder_out.shape[1]
        inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_out,
            "use_cache_branch": np.array([False], dtype=bool)
        }
        for i in range(4):
            inputs[f"past_key_values.{i}.decoder.key"] = np.zeros((batch_size, num_heads, 0, head_dim), dtype=np.float32)
            inputs[f"past_key_values.{i}.decoder.value"] = np.zeros((batch_size, num_heads, 0, head_dim), dtype=np.float32)
            inputs[f"past_key_values.{i}.encoder.key"] = np.zeros((batch_size, num_heads, encoder_seq_len, head_dim), dtype=np.float32)
            inputs[f"past_key_values.{i}.encoder.value"] = np.zeros((batch_size, num_heads, encoder_seq_len, head_dim), dtype=np.float32)
        try:
            outputs = decoder_sess.run(None, inputs)
            logits = outputs[0]
            temperature = 0.1
            logits = logits[0, -1, :] / temperature
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            next_token = int(np.argmax(probs))
        except Exception as e:
            print(f"Decoder error at step {step}: {e}")
            break
        tokens.append(next_token)
        if next_token == special_tokens.get("<|endoftext|>", 50256):
            break
        if next_token == special_tokens.get("<|nospeech|>", 50361):
            break
    content_tokens = [t for t in tokens[4:] if t < 50000]
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
