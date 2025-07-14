
from __future__ import annotations
from PIL import Image
import cv2
import numpy as np
import torch
from pathlib import Path

# Suppress CUDA warnings from EasyOCR and other libraries
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import onnxruntime as ort
import os
import numpy as np
from easyocr.easyocr import Reader

from pathlib import Path
from tokenizers import Tokenizer

# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import Callable

import cv2
import numpy as np
import re
import torch
import torch.nn.functional as F
from easyocr.craft_utils import adjustResultCoordinates, getDetBoxes
from easyocr.easyocr import Reader
from easyocr.imgproc import normalizeMeanVariance, resize_aspect_ratio
from easyocr.recognition import AlignCollate, ListDataset, custom_mean
from easyocr.utils import (
    diff,
    get_image_list,
    group_text_box,
    make_rotated_img_list,
    reformat_input,
    set_result_with_confidence,
)
from easyocr.utils import CTCLabelConverter
from PIL import Image
from torch.utils.data import DataLoader

from qai_hub_models.utils.draw import draw_box_from_xyxy
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

DETECTOR_ARGS = {
    "canvas_size": 2560,
    "mag_ratio": 1.0,
    "estimate_num_chars": False,
    "text_threshold": 0.7,
    "link_threshold": 0.4,
    "low_text": 0.4,
    "poly": False,
    "estimate_num_chars": False,
    "optimal_num_chars": None,
    "slope_ths": 0.1,
    "ycenter_ths": 0.5,
    "height_ths": 0.5,
    "width_ths": 0.5,
    "add_margin": 0.1,
    "min_size": 20,
}

RECOGNIZER_ARGS = {
    "allowlist": None,
    "blocklist": None,
    "beamWidth": 5,
    "detail": 1,
    "rotation_info": None,
    "contrast_ths": 0.1,
    "adjust_contrast": 0.5,
    "filter_ths": 0.003,
}


class EasyOCRAppDemo:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with EasyOCR.

    The app uses 2 models:
        * detector
        * recognizer

    For a given image input, the app will:
        * use the reformat_input() to pre-process the image (img, img_cv_grey)
        * run detector_preprocess() change input to tensor
        * run detector and post-process the result to bbox list
        * run recognizer twice for low confident score
        * map the score according to decoder
    """

    def __init__(
        self,
        detector: Callable[[torch.Tensor], torch.Tensor],
        recognizer: Callable[[torch.Tensor], torch.Tensor],
        lang_list,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.imgH = 64
        self.decoder = "greedy"
        ocr_reader = Reader(
            lang_list,
            gpu=False,
            quantize=False,
        )
        self.character = ocr_reader.character
        self.lang_char = ocr_reader.lang_char
        self.model_lang = ocr_reader.model_lang
        self.converter = ocr_reader.converter

    def detector_preprocess(self, img: np.ndarray):
        image_arrs = [img]

        img_resized_list = []
        # resize
        for img in image_arrs:
            img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
                img,
                DETECTOR_ARGS["canvas_size"],
                interpolation=cv2.INTER_LINEAR,
                mag_ratio=DETECTOR_ARGS["mag_ratio"],
            )
            img_resized_list.append(img_resized)

        ratio_h = ratio_w = 1 / target_ratio
        # preprocessing
        x = [
            np.transpose(normalizeMeanVariance(n_img), (2, 0, 1))
            for n_img in img_resized_list
        ]
        x = torch.from_numpy(np.array(x))

        return x, (ratio_h, ratio_w)

    def detector_postprocess(self, results: torch.Tensor, infos: torch.Tensor):
        ratio_w = infos[1]
        ratio_h = infos[0]
        result, horizontal_list_agg, free_list_agg = [], [], []

        boxes_list, polys_list = [], []
        for out in results:
            # make score and link map
            score_text = out[:, :, 0].cpu().data.numpy()
            score_link = out[:, :, 1].cpu().data.numpy()

            # Post-processing
            required_keys = [
                "textmap",
                "linkmap",
                "text_threshold",
                "link_threshold",
                "low_text",
                "poly",
                "estimate_num_chars",
            ]
            filtered_args = {
                k: DETECTOR_ARGS[k] for k in required_keys if k in DETECTOR_ARGS
            }

            boxes, polys, mapper = getDetBoxes(score_text, score_link, **filtered_args)

            # coordinate adjustment
            boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
            polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
            if DETECTOR_ARGS["estimate_num_chars"]:
                boxes = list(boxes)
                polys = list(polys)
            for k in range(len(polys)):
                if DETECTOR_ARGS["estimate_num_chars"]:
                    boxes[k] = (boxes[k], mapper[k])
                if polys[k] is None:
                    polys[k] = boxes[k]
            boxes_list.append(boxes)
            polys_list.append(polys)

        if DETECTOR_ARGS["estimate_num_chars"]:
            polys_list = [
                [
                    p
                    for p, _ in sorted(
                        polys,
                        key=lambda x: abs(DETECTOR_ARGS["optimal_num_chars"] - x[1]),
                    )
                ]
                for polys in polys_list
            ]

        for polys in polys_list:
            single_img_result = []
            for i, box in enumerate(polys):
                poly = np.array(box).astype(np.int32).reshape(-1)
                single_img_result.append(poly)
            result.append(single_img_result)

        required_keys = [
            "slope_ths",
            "ycenter_ths",
            "height_ths",
            "width_ths",
            "add_margin",
        ]
        filtered_args = {
            k: DETECTOR_ARGS[k] for k in required_keys if k in DETECTOR_ARGS
        }
        for text_box in result:

            horizontal_list, free_list = group_text_box(text_box, **filtered_args)
            if DETECTOR_ARGS["min_size"]:
                horizontal_list = [
                    i
                    for i in horizontal_list
                    if max(i[1] - i[0], i[3] - i[2]) > DETECTOR_ARGS["min_size"]
                ]
                free_list = [
                    i
                    for i in free_list
                    if max(diff([c[0] for c in i]), diff([c[1] for c in i]))
                    > DETECTOR_ARGS["min_size"]
                ]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)

        return horizontal_list_agg, free_list_agg

    def recognizer_preprocess(
        self,
        img_cv_grey: np.ndarray,
        horizontal_list: list[list[int]] | None,
        free_list: list[list[int]] | None,
        batch_size: int,
    ):
        if RECOGNIZER_ARGS["allowlist"]:
            assert isinstance(RECOGNIZER_ARGS["allowlist"], str)
            ignore_char = "".join(
                set(self.character) - set(RECOGNIZER_ARGS["allowlist"])
            )
        elif RECOGNIZER_ARGS["blocklist"]:
            assert isinstance(RECOGNIZER_ARGS["blocklist"], str)
            ignore_char = "".join(set(RECOGNIZER_ARGS["blocklist"]))
        else:
            ignore_char = "".join(set(self.character) - set(self.lang_char))

        if self.model_lang in ["chinese_tra", "chinese_sim"]:
            self.decoder = "greedy"

        if (horizontal_list is None) and (free_list is None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        return img_cv_grey, horizontal_list, free_list, ignore_char, batch_size

    def recognize(
        self,
        img_cv_grey: np.ndarray,
        horizontal_list: list[list[int]],
        free_list: list[list[int]],
        batch_size: int,
        ignore_char: str,
    ):
        if batch_size == 1 and not RECOGNIZER_ARGS["rotation_info"]:
            result = []
            for bbox in horizontal_list:
                h_list = [bbox]
                f_list: list[list[int]] = []
                image_list, max_width = get_image_list(
                    h_list, f_list, img_cv_grey, model_height=self.imgH
                )
                result0 = self.recognizer_get_text(
                    int(max_width), image_list, ignore_char, batch_size
                )
                result += result0
            for bbox in free_list:
                h_list = []
                f_list = [bbox]
                image_list, max_width = get_image_list(
                    h_list, f_list, img_cv_grey, model_height=self.imgH
                )
                result0 = self.recognizer_get_text(
                    int(max_width), image_list, ignore_char, batch_size
                )
                result += result0
        # default mode will try to process multiple boxes at the same time
        else:
            image_list, max_width = get_image_list(
                horizontal_list, free_list, img_cv_grey, model_height=self.imgH
            )
            image_len = len(image_list)
            if RECOGNIZER_ARGS["rotation_info"] and image_list:
                image_list = make_rotated_img_list(
                    RECOGNIZER_ARGS["rotation_info"], image_list
                )
                max_width = max(max_width, self.imgH)

            result = self.recognizer_get_text(
                int(max_width), image_list, ignore_char, batch_size
            )

            if RECOGNIZER_ARGS["rotation_info"] and (horizontal_list + free_list):
                # Reshape result to be a list of lists, each row being for
                # one of the rotations (first row being no rotation)
                result = set_result_with_confidence(
                    [
                        result[image_len * i : image_len * (i + 1)]
                        for i in range(len(RECOGNIZER_ARGS["rotation_info"]) + 1)  # type: ignore[arg-type]
                    ]
                )
        return result

    def recognizer_get_text(
        self,
        imgW: int,
        image_list: list[np.ndarray],
        ignore_char: str = "",
        batch_size: int = 1,
        workers: int = 1,
    ):
        ignore_idx = []
        for char in ignore_char:
            ignore_idx.append(self.character.index(char) + 1)

        coord: list[np.ndarray] = [item[0] for item in image_list]
        img_list = [item[1] for item in image_list]
        AlignCollate_normal = AlignCollate(
            imgH=self.imgH, imgW=imgW, keep_ratio_with_pad=True
        )
        test_data = ListDataset(img_list)
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(workers),
            collate_fn=AlignCollate_normal,
            pin_memory=False,
        )

        # predict first round
        result1 = self.recognizer_inference(test_loader, ignore_idx)

        # predict second round
        low_confident_idx = [
            i
            for i, item in enumerate(result1)
            if (item[1] < RECOGNIZER_ARGS["contrast_ths"])
        ]

        result2 = []
        if len(low_confident_idx) > 0:
            img_list2 = [img_list[i] for i in low_confident_idx]
            AlignCollate_contrast = AlignCollate(
                imgH=self.imgH,
                imgW=imgW,
                keep_ratio_with_pad=True,
                adjust_contrast=RECOGNIZER_ARGS["adjust_contrast"],
            )
            test_data = ListDataset(img_list2)
            test_loader = DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(workers),
                collate_fn=AlignCollate_contrast,
                pin_memory=False,
            )
            result2 = self.recognizer_inference(test_loader, ignore_idx)

        result = []
        for i, zipped in enumerate(zip(coord, result1)):
            box, pred1 = zipped
            if i in low_confident_idx:
                pred2 = result2[low_confident_idx.index(i)]
                if pred1[1] > pred2[1]:
                    result.append((box, pred1[0], pred1[1]))
                else:
                    result.append((box, pred2[0], pred2[1]))
            else:
                result.append((box, pred1[0], pred1[1]))

        return result

    def recognizer_inference(
        self, test_loader: DataLoader, ignore_idx: list[int], device="cpu"
    ):
        result = []
        with torch.no_grad():
            for image_tensors in test_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                preds = self.recognizer(image)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)

                ######## filter ignore_char, rebalance
                preds_prob = F.softmax(preds, dim=2)
                preds_prob[:, :, ignore_idx] = 0.0
                pred_norm = preds_prob.sum(dim=2)
                preds_prob = preds_prob / pred_norm.unsqueeze(-1)

                preds_str: list[str]
                if self.decoder == "greedy":
                    # Select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds_prob.max(2)
                    preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode_greedy(
                        preds_index.data.cpu().detach().numpy(), preds_size.data
                    )
                elif self.decoder == "beamsearch":
                    k = preds_prob.cpu().detach().numpy()
                    preds_str = self.converter.decode_beamsearch(
                        k, beamWidth=RECOGNIZER_ARGS["beamWidth"]
                    )
                elif self.decoder == "wordbeamsearch":
                    k = preds_prob.cpu().detach().numpy()
                    preds_str = self.converter.decode_wordbeamsearch(
                        k, beamWidth=RECOGNIZER_ARGS["beamWidth"]
                    )
                else:
                    raise NotImplementedError(f"Unknown decoder {self.decoder}")

                preds_prob_np = preds_prob.cpu().detach().numpy()
                values = preds_prob_np.max(axis=2)
                indices = preds_prob_np.argmax(axis=2)
                preds_max_prob = []
                for v, i in zip(values, indices):
                    max_probs = v[i != 0]
                    if len(max_probs) > 0:
                        preds_max_prob.append(max_probs)
                    else:
                        preds_max_prob.append(np.array([0]))

                for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                    confidence_score = custom_mean(pred_max_prob)
                    result.append([pred, confidence_score])

        return result

    def predict(self, *args, **kwargs):
        return self.predict_text_from_image(*args, **kwargs)

    def predict_text_from_image(self, pixel_values_or_image: np.ndarray | Image.Image):
        """
        From provided array or image, predict (bounding box of texts, text, scores)

        Parameters:
            pixel_values_or_image
                PIL image(s)
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout

        Returns:
            results: tuple[Image.Image, list[tuple]]
                Image.Image: it will be origin image with bounding box
                list[tuple]: list of tuple(bounding box coords, text, scores)
        """
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(pixel_values_or_image)
        batch_size = len(NHWC_int_numpy_frames)
        # TODO
        NHWC_int_numpy_frame = NHWC_int_numpy_frames[0]

        img, img_cv_grey = reformat_input(NHWC_int_numpy_frame)

        # detector
        input_tensor, infos = self.detector_preprocess(img)
        results, feature = self.detector(input_tensor)
        horizontal_list, free_list = self.detector_postprocess(results, infos)

        # recognizer
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        (
            img_cv_grey,
            horizontal_list,
            free_list,
            ignore_char,
            batch_size,
        ) = self.recognizer_preprocess(
            img_cv_grey, horizontal_list, free_list, batch_size
        )
        list_result = self.recognize(
            img_cv_grey, horizontal_list, free_list, batch_size, ignore_char
        )

        coords = [item[0] for item in list_result]
        # for coord in coords:
        #     draw_box_from_xyxy(
        #         NHWC_int_numpy_frame,
        #         tuple(coord[0]),
        #         tuple(coord[2]),
        #         color=(0, 255, 0),
        #         size=2,
        #     )

        print(list_result)
        return (Image.fromarray(NHWC_int_numpy_frames[0]), list_result)


class EasyOCRApp_ort:
    def __init__(self, detector_session: ort.InferenceSession, recognizer_session: ort.InferenceSession):
        self.detector_session = detector_session
        self.recognizer_session = recognizer_session
        self.ocr_reader = Reader(
            ["en"],
            gpu=False,
            quantize=False,
        )
        self.converter = self.ocr_reader.converter

    def setup_sessions(self):
        detector_model_path = "backend/ocr/models/detector.onnx"
        recognizer_model_path = "backend/ocr/models/recognizer.onnx"

        self.detector_session = ort.InferenceSession(self.detector_model_path)
        self.recognizer_session = ort.InferenceSession(self.recognizer_model_path)

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
        thresh = (score_map > 0.1).astype(np.uint8) * 255
        print("Score map stats:")
        print(f"min: {score_map.min()}, max: {score_map.max()}, mean: {score_map.mean()}")
        score_map_vis = (score_map * 255).clip(0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(score_map_vis, cv2.COLORMAP_JET)

        # Save or display the heatmap
        cv2.imwrite("./backend/ocr/scratch_data/score_map_heatmap.png", heatmap)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))  # Wider than tall to group lines
        # dilated = cv2.dilate(thresh, kernel, iterations=2)
        # cv2.imwrite("./backend/ocr/scratch_data/dilated.png", dilated)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_debug = heatmap.copy()
        cv2.drawContours(contour_debug, contours, -1, (0, 255, 0), 1)
        cv2.imwrite("./backend/ocr/scratch_data/contours_debug.png", contour_debug)
        
        input_h, input_w = score_map.shape
        orig_w, orig_h = orig_image.size

        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes = []
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            if w > 10 and h > 5:
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


    def manual_decode(self, preds_index, char_list, blank_idx=0):
        decoded = []
        prev_idx = None
        preds_index = np.ravel(preds_index)

        for idx in preds_index:
            idx = int(idx)
            if idx != blank_idx and idx != prev_idx:
                if idx < len(char_list):
                    decoded.append(char_list[idx])
            prev_idx = idx
        return ''.join(decoded)

    # decode recognizer output logits to text using basic greedy CTC decoding
    def recognizer_postprocess(self, recognizer_output):
        logits = recognizer_output[0]  # shape [1, seq_len, vocab_size]
        preds_index = np.argmax(logits, axis=2)[0]  # shape [seq_len]

        # preds_size: length of the sequence (assuming full length here)
        preds_size = np.array([logits.shape[1]])
        text = self.converter.decode_greedy(preds_index, preds_size)[0]
        return text

    def write_text_to_file(self, text, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

    # runs recognizer onnx model
    def recognizer_inference(self, input_tensor: np.ndarray):
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

    def extract_recognizable_words(self, raw_text: str, min_len: int = 3):
        words = re.findall(r'\b[\w-]{%d,}\b' % min_len, raw_text)
        filtered = []

        for word in words:
            # Drop all-uppercase gibberish words like 'XCCF', '4EeTe~', etc.
            if word.isupper() and not word.isalpha():
                continue
            # Drop mostly symbols
            if re.fullmatch(r'[^a-zA-Z0-9]+', word):
                continue
            # Drop words with more punctuation than letters
            if sum(c.isalpha() for c in word) < len(word) / 2:
                continue
            filtered.append(word)

        return filtered