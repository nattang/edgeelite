# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# from transformers.utils.import_utils import DETECTRON2_IMPORT_ERROR
from app import  EasyOCRApp_ort
from trocr_app import TrocrApp
# ort imports
import onnxruntime as ort
import os
import numpy as np
from pathlib import Path
from PIL import Image
from easyocr.easyocr import Reader

from transformers import TrOCRProcessor

def run_easyocr(image_path: str):
    base_dir = Path(__file__).parent.parent    
    detector_model_path = base_dir / "models" / "easyocr-easyocrdetector.onnx"
    recognizer_model_path = base_dir / "models" / "easyocr-easyocrrecognizer.onnx"
 
    onnxruntime_dir = Path(ort.__file__).parent

    hexagon_driver = onnxruntime_dir / "capi" / "QnnHtp.dll"

    session_options = ort.SessionOptions()
    qnn_provider_options = {
        "backend_path": hexagon_driver
    }

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.log_severity_level = 3

    detector_session = ort.InferenceSession(detector_model_path, 
                                providers= [("QNNExecutionProvider",qnn_provider_options),"CPUExecutionProvider"],
                                sess_options=so
                                )
    detector_session.get_providers()
    print("Detector session input names: ")
    for inp in detector_session.get_inputs():
        print(inp.name, inp.shape, inp.type)

    print("Detector session output names: ")
    for out in detector_session.get_outputs():
        print(out.name, out.shape, out.type)

    recognizer_session = ort.InferenceSession(recognizer_model_path, 
                                providers= [("QNNExecutionProvider",qnn_provider_options),"CPUExecutionProvider"],
                                sess_options=so
                                )
    recognizer_session.get_providers()
    print("Recognizer session input names: ")
    for inp in recognizer_session.get_inputs():
        print(inp.name, inp.shape, inp.type)

    print("Recognizer session output names: ")
    for out in recognizer_session.get_outputs():
        print(out.name, out.shape, out.type)

    ocr = EasyOCRApp_ort(detector_session, recognizer_session)
    original_image = Image.open(image_path).convert("RGB")
    processed_image = ocr.detector_preprocess(image_path)

    detector_outputs = ocr.detector_inference(processed_image)
    boxes = ocr.detector_postprocess(detector_outputs, original_image)
    ocr.draw_boxes_on_image(original_image, boxes, "./backend/ocr/scratch_data/image_with_boxes.jpg")
    print("boxes: ", boxes)

    recognized_texts = []
    reader = Reader(['en'])
    char_list = reader.character
    char_list = list(char_list)

    for box in boxes:
        region_input = ocr.crop_and_prepare_region(original_image, box)
        recognizer_outputs = recognizer_session.run(None, {"image": region_input})
        
        # TODO: look into actual char list -- could be wrong right now
        text = ocr.recognizer_postprocess(recognizer_outputs, char_list)
        recognized_texts.append(text)

    full_text = "\n".join(recognized_texts)
    print(full_text)

    # TODO: put in database
    ocr.write_text_to_file(full_text, "./backend/ocr/scratch_data/recognized_text.txt")

def setup_trocr_sessions():
    base_dir = Path(__file__).parent.parent
    encoder_model_path = base_dir / "models" / "ocr" / "trocr-trocrencoder.onnx"
    decoder_model_path = base_dir / "models" / "ocr" / "trocr-trocrdecoder.onnx"

    root_dir = Path.cwd().parent.parent
    onnxruntime_dir = Path(ort.__file__).parent
    hexagon_driver = onnxruntime_dir / "capi" / "QnnHtp.dll"

    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.log_severity_level = 3

    qnn_provider_options = {
        "backend_path": hexagon_driver
    }

    encoder_session = ort.InferenceSession(
        encoder_model_path,
        providers=[("QNNExecutionProvider", qnn_provider_options), "CPUExecutionProvider"],
        sess_options=session_options
    )

    decoder_session = ort.InferenceSession(
        decoder_model_path,
        providers=[("QNNExecutionProvider", qnn_provider_options), "CPUExecutionProvider"],
        sess_options=session_options
    )

    print("Decoder session output names:")
    for out in decoder_session.get_outputs():
        print(out.name, out.shape, out.type)

    # print("Encoder session providers:")
    # print(encoder_session.get_providers())
    # print("Decoder session providers:")
    # print(decoder_session.get_providers())
    return encoder_session, decoder_session

def run_trocr(image_path: str):
    encoder_session, decoder_session = setup_trocr_sessions()
    trocr = TrocrApp(encoder_session, decoder_session)
    processed_image = trocr.process_image(image_path)
    encoder_outputs = encoder_session.run(None, {"pixel_values": processed_image})
    print("Finished encoder inference")
    for i, out in enumerate(encoder_outputs):
        print(f"Encoder output {i} stats: min={out.min()}, max={out.max()}, mean={out.mean()}")

    decoder_start_token = 0
    eos_token = 2
    num_layers = 6
    num_heads = 8
    kv_dim = 32

    # Initial decoder input
    decoder_inputs = trocr.encoder_to_decoder_cache(
        encoder_outputs,
        decoder_start_token,
        num_layers=num_layers,
        num_heads=num_heads,
        kv_dim=kv_dim
    )

    output_tokens = [decoder_start_token]
    cur_index = 0

    # while True:
    print("running decoder session")
    # trocr.check_decoder_input_shapes(decoder_inputs, decoder_session)
    outputs = decoder_session.run(None, decoder_inputs)
    output_tokens.append(outputs[0])
    print("output_tokens: ", output_tokens)

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
    # text = processor.decode(output_tokens, skip_special_tokens=True)
    # print("text: ", text)

def process_image(image_path: str):
    run_easyocr(image_path)
    # run_trocr(image_path)
