# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# from transformers.utils.import_utils import DETECTRON2_IMPORT_ERROR
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from .app import EasyOCRAppDemo, EasyOCRApp_ort
from .easyocr_app import EasyOCRApp
from .trocr_app import TrocrApp
# ort imports
import onnxruntime as ort
import os
import numpy as np
from pathlib import Path
from PIL import Image
from easyocr.easyocr import Reader

from transformers import TrOCRProcessor

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

def run_easyocr(image_path: str):
    print("Running EasyOCR")
    base_dir = Path(__file__).parent.parent    
    detector_model_path = base_dir / "models" / "ocr" / "easyocr-easyocrdetector.onnx"
    recognizer_model_path = base_dir / "models" / "ocr" / "easyocr-easyocrrecognizer.onnx"
 
    # Check if ONNX models exist
    if not detector_model_path.exists() or not recognizer_model_path.exists():
        print("⚠️ ONNX OCR models not found, falling back to basic EasyOCR")
        return run_easyocr_basic(image_path)
    
    onnxruntime_dir = Path(ort.__file__).parent

    hexagon_driver = onnxruntime_dir / "capi" / "QnnHtp.dll"

    qnn_provider_options = {
        "backend_path": hexagon_driver
    }

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.log_severity_level = 3

    try:
        # Check if QNN is available
        available_providers = ort.get_available_providers()
        if 'QNNExecutionProvider' in available_providers:
            print("🚀 Using QNN NPU for OCR")
            providers = [("QNNExecutionProvider", qnn_provider_options), "CPUExecutionProvider"]
        else:
            print("⚠️ QNN not available, using CPU for OCR")
            providers = ["CPUExecutionProvider"]
        
        detector_session = ort.InferenceSession(detector_model_path, 
                                    providers=providers,
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
                                    providers=providers,
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
        # ocr.write_text_to_file(full_text, "./backend/ocr/scratch_data/recognized_text.txt")
        return full_text
        
    except Exception as e:
        print(f"⚠️ ONNX OCR failed: {e}")
        print("🔄 Falling back to basic EasyOCR")
        return run_easyocr_basic(image_path)

def run_easyocr_basic(image_path: str):
    """Basic EasyOCR implementation without ONNX models."""
    try:
        from easyocr import Reader
        print("🔄 Running basic EasyOCR (CPU)")
        
        # Initialize EasyOCR reader
        reader = Reader(['en'])
        
        # Read the image
        result = reader.readtext(image_path)
        
        # Extract text from results
        texts = []
        for detection in result:
            text = detection[1]  # The detected text
            confidence = detection[2]  # Confidence score
            if confidence > 0.5:  # Only include high-confidence detections
                texts.append(text)
        
        full_text = "\n".join(texts)
        print(f"✅ OCR Result: {full_text}")
        return full_text
        
    except Exception as e:
        print(f"❌ Basic EasyOCR failed: {e}")
        return "OCR processing failed"

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
    reader = Reader(['en'])
    char_list = reader.character
    char_list = list(char_list)

    print("Number of boxes: ", len(boxes))
    for box in boxes:
        region_input = ocr.crop_and_prepare_region(original_image, box)
        print("running recognizer session")
        recognizer_outputs = recognizer_session.run(None, {"image": region_input})
        
        # TODO: look into actual char list -- could be wrong right now
        text = ocr.recognizer_postprocess(recognizer_outputs, char_list)
        recognized_texts.append(text)

    full_text = "\n".join(recognized_texts)
    # full_text = ocr.extract_recognizable_words(full_text)
    print("results: ", full_text)
    return full_text

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

def run_easyocr_ort_test(image_path: str):
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

    ocr = EasyOCRApp(detector_session, recognizer_session, ['en'])
    # image = Image.open(image_path).convert("RGB")
    # processed_image = ocr.preprocess_image(image_path)
    results = ocr.predict_text_from_image(image_path)
    print("results: ", results)
    return results


def run_easyocr_demo(image_path: str):
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
    """Process image with OCR - tries ONNX models first, falls back to basic EasyOCR."""
    try:
        # Try ONNX EasyOCR first
        print(f"🖼️ Processing image with ONNX EasyOCR: {image_path}")
        result = run_easyocr(image_path)
        if result and result != "Error":
            return result
    except Exception as e:
        print(f"⚠️ ONNX OCR failed: {e}")
    
    try:
        # Fallback to basic EasyOCR
        print(f"🔄 Falling back to basic EasyOCR")
        return run_easyocr_basic(image_path)
    except Exception as e:
        print(f"❌ Basic OCR also failed: {e}")
        return f"OCR processing failed: {str(e)}"

# if __name__ == "__main__":
    # run_easyocr_ort_test(r"C:\Users\HAQKATHON SCL\Downloads\maxresdefault.jpg")
    # run_easyocr_ort_test(r"C:\Users\HAQKATHON SCL\Pictures\Screenshots\Screenshot 2025-07-11 170005.png")

    # run_easyocr_demo(r"C:\Users\HAQKATHON SCL\Downloads\maxresdefault.jpg")
    # run_easyocr_demo(r"C:\Users\HAQKATHON SCL\Pictures\Screenshots\Screenshot 2025-07-11 170005.png")
    # run_easyocr_ort_test(r"C:\Users\HAQKATHON SCL\Downloads\maxresdefault.jpg")
    # run_easyocr_ort(r"C:\Users\HAQKATHON SCL\Downloads\maxresdefault.jpg")
    # run_easyocr_ort(r"C:\Users\HAQKATHON SCL\Pictures\Screenshots\Screenshot 2025-07-11 170005.png")