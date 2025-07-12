from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)
model = whisper.load_model("base") 

@app.route('/asr', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    audio_file = request.files['file']
    path = 'temp.wav'
    audio_file.save(path)

    result = model.transcribe(path)
    os.remove(path)
    return jsonify({'text': result['text']})

if __name__ == '__main__':
    app.run(port=5000)
