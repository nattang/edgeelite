# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# from transformers.utils.import_utils import DETECTRON2_IMPORT_ERROR
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from backend.ocr.app import EasyOCRAppDemo, EasyOCRApp_ort

# ort imports
import onnxruntime as ort
import os
import numpy as np
from pathlib import Path
from PIL import Image
from easyocr.easyocr import Reader

# demo imports 
# from qai_hub_models.models.easyocr.app import EasyOCRApp
from qai_hub_models.models.easyocr.model import MODEL_ASSET_VERSION, MODEL_ID, EasyOCR
from qai_hub_models.utils.args import (
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image

def run_easyocr_ort(image_path: str):
    print("Running EasyOCR")
    base_dir = Path(__file__).parent.parent    
    detector_model_path = base_dir / "models" / "ocr" / "easyocr-easyocrdetector.onnx"
    recognizer_model_path = base_dir / "models" / "ocr" / "easyocr-easyocrrecognizer.onnx"

    onnxruntime_dir = Path(ort.__file__).parent
    hexagon_driver = onnxruntime_dir / "capi" / "QnnHtp.dll"

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

    recognized_texts = []

    print("Number of boxes: ", len(boxes))
    for box in boxes:
        region_input = ocr.crop_and_prepare_region(original_image, box)
        print("running recognizer session")
        recognizer_outputs = recognizer_session.run(None, {"image": region_input})
        
        text = ocr.recognizer_postprocess(recognizer_outputs)
        recognized_texts.append(text)
        

    full_text = "\n".join(recognized_texts)
    # full_text = ocr.extract_recognizable_words(full_text)
    # print("results: ", recognized_texts)
    print("full_text: ", full_text)
    return full_text


    # Load app and image
    parser = get_model_cli_parser(EasyOCR)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    args = parser.parse_args(None)
    image = load_image(image_path)
    model = model_from_cli_args(EasyOCR, args)
    app = EasyOCRAppDemo(model.detector, model.recognizer, model.lang_list)
    print("Model Loaded")

    results = app.predict_text_from_image(image)

def process_image(image_path: str):
    # run_easyocr(image_path)
    return run_easyocr_ort(image_path)
    # run_trocr(image_path)

# if __name__ == "__main__":
    # run_easyocr_ort_test(r"C:\Users\HAQKATHON SCL\Downloads\maxresdefault.jpg")
    # run_easyocr_ort_test(r"C:\Users\HAQKATHON SCL\Pictures\Screenshots\Screenshot 2025-07-11 170005.png")

    # run_easyocr_demo(r"C:\Users\HAQKATHON SCL\Downloads\maxresdefault.jpg")
    # run_easyocr_demo(r"C:\Users\HAQKATHON SCL\Pictures\Screenshots\Screenshot 2025-07-11 170005.png")
    # run_easyocr_ort_test(r"C:\Users\HAQKATHON SCL\Downloads\maxresdefault.jpg")
    # run_easyocr_ort(r"C:\Users\HAQKATHON SCL\Downloads\maxresdefault.jpg")
    # run_easyocr_ort(r"C:\Users\HAQKATHON SCL\Pictures\Screenshots\Screenshot 2025-07-11 170005.png")