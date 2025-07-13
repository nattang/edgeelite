import onnxruntime as ort

class TrocrApp:
    def __init__(self, encoder_session: ort.InferenceSession, decoder_session: ort.InferenceSession):
        self.encoder_session = encoder_session
        self.decoder_session = decoder_session

    def encoder_preprocess(self, image_path: str):
        pass

    def decoder_preprocess(self, image_path: str):
        pass