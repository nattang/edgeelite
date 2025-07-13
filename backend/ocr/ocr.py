# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# from transformers.utils.import_utils import DETECTRON2_IMPORT_ERROR
from app import  EasyOCRApp_ort
# ort imports
import onnxruntime as ort
import os
import numpy as np
from pathlib import Path
from PIL import Image
from easyocr.easyocr import Reader

def run_ocr(image_path: str):
    base_dir = Path(__file__).parent    
    detector_model_path = base_dir / "models" / "easyocr-easyocrdetector.onnx"
    recognizer_model_path = base_dir / "models" / "easyocr-easyocrrecognizer.onnx"
    
    root_dir = Path.cwd().parent.parent 
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
    

def process_image(image_path: str):
    run_ocr(image_path)

if __name__ == "__main__":
    process_image(r"C:\Users\HAQKATHON SCL\Pictures\Screenshots\Screenshot 2025-07-11 170005.png")
