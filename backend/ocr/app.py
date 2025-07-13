
from __future__ import annotations
from PIL import Image
import cv2
import numpy as np
import torch
from pathlib import Path

import onnxruntime as ort
import os
import numpy as np
from easyocr.easyocr import Reader

from pathlib import Path
from tokenizers import Tokenizer


class EasyOCRApp_ort:
    def __init__(self, detector_session: ort.InferenceSession, recognizer_session: ort.InferenceSession):
        self.detector_session = detector_session
        self.recognizer_session = recognizer_session

    # convert image to numpy array suitable for detector input
    def detector_preprocess(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((800, 608))  
        img_np = np.asarray(img).astype(np.float32)
        img_np = img_np.transpose(2, 0, 1)
        img_np /= 255.0
        img_np = np.expand_dims(img_np, axis=0)
        return img_np

    # runs detector onnx model
    def detector_inference(self, input_arr):
        outputs = self.detector_session.run(None, {"image": input_arr})
        return outputs

    # returns list of bounding boxes in original image coordinates
    def detector_postprocess(self, detector_outputs, orig_image):
        results = detector_outputs[0]  # (1,304,400,2)
        score_map = results[0, :, :, 0]

        # Threshold score map
        thresh = (score_map > 0.05).astype(np.uint8) * 255
        print("Score map stats:")
        print(f"min: {score_map.min()}, max: {score_map.max()}, mean: {score_map.mean()}")
        print("Threshold mask unique values:", np.unique(thresh))
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        input_h, input_w = score_map.shape
        orig_w, orig_h = orig_image.size

        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes = []
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            if w > 10 and h > 10:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                boxes.append((x1, y1, x2, y2))

        return boxes

    # crop and prepare single text region for recognizer input
    def crop_and_prepare_region(self, image: Image, box, padding: int = 5):
        x1, y1, x2, y2 = box

        # Add padding and ensure box stays within image bounds
        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, image.width)
        y2 = min(y2 + padding, image.height)

        region = image.crop((x1, y1, x2, y2)).convert("L")  # grayscale
        region = region.resize((1000, 64), Image.BILINEAR)  # recognizer input size
        region_np = np.array(region).astype(np.float32) / 255.0
        region_np = np.expand_dims(region_np, axis=0)  # channel
        region_np = np.expand_dims(region_np, axis=0)  # batch
        return region_np

    # decode recognizer output logits to text using basic greedy CTC decoding
    def recognizer_postprocess(self, recognizer_output, char_list, blank_idx=0):
        logits = recognizer_output[0]  # shape [1, seq_len, vocab_size]
        preds_index = np.argmax(logits, axis=2)[0]  # shape [seq_len]

        # preds_size: length of the sequence (assuming full length here)
        preds_size = np.array([logits.shape[1]])

        # Use EasyOCR's converter to decode CTC output
        ocr_reader = Reader(
            ['en'],
            gpu=True,
            quantize=False,
        )
        text = ocr_reader.converter.decode_greedy(preds_index, preds_size)[0]
        return text

    def write_text_to_file(self, text, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

    # runs recognizer onnx model
    def recognizer_inference(self, input_tensor: torch.Tensor):
        return self.recognizer_session.run(None, {"input": input_tensor})

    def draw_boxes_on_image(self, image: Image.Image, boxes: list[tuple[int, int, int, int]], output_path: str):
        """
        Draw bounding boxes on the image and save the result.

        Args:
            image: PIL.Image (RGB)
            boxes: List of (x1, y1, x2, y2) bounding boxes
            output_path: Path to save the image with boxes
        """
        # Convert PIL to OpenCV format
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save the image
        cv2.imwrite(output_path, image_cv)
        print(f"Saved image with boxes to: {output_path}")