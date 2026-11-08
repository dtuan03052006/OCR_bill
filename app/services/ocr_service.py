from typing import List, Tuple
import cv2
import easyocr
import numpy as np
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

from app.config import settings


class OCRService:
    def __init__(self):
        # 1. Khởi tạo EasyOCR detector
        self.detector = easyocr.Reader(
            ["en"],
            gpu=(settings.DEVICE == "cuda"),
            recognizer=False,
            model_storage_directory="weights/easyocr",
            user_network_directory="weights/easyocr"
        )

        # 2. Khởi tạo VietOCR recognizer
        v_cfg = Cfg.load_config_from_file(settings.VIETOCR_CONFIG)
        v_cfg["weights"] = settings.VIETOCR_WEIGHTS
        v_cfg["device"] = settings.DEVICE
        v_cfg["predictor"]["beamsearch"] = False
        self.recognizer = Predictor(v_cfg)
        print("[+] Khởi tạo OCRService thành công!")

    @staticmethod
    def sort_reading_order_boxes(boxes: List[List[int]]) -> List[List[int]]:
        """Gom cụm và sắp xếp các bounding boxes theo thứ tự đọc tự nhiên"""
        if not boxes:
            return []
        heights = [(b[3] - b[1]) for b in boxes]
        avg_h = np.median(heights) if heights else 20
        y_threshold = max(avg_h * 0.5, 10)

        # Sắp xếp sơ bộ theo Y
        sorted_y = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
        lines = []
        cur_line = [sorted_y[0]]

        for b in sorted_y[1:]:
            b_cy = (b[1] + b[3]) / 2
            line_cy = np.mean([(x[1] + x[3]) / 2 for x in cur_line])
            if abs(b_cy - line_cy) <= y_threshold:
                cur_line.append(b)
            else:
                lines.append(sorted(cur_line, key=lambda x: x[0]))
                cur_line = [b]
        if cur_line:
            lines.append(sorted(cur_line, key=lambda x: x[0]))

        return [b for line in lines for b in line]

    def extract_text(self, image_bgr: np.ndarray, image_pil: Image.Image) -> Tuple[List[List[int]], List[str]]:
        """
        Nhận vào ảnh, trả về danh sách:
        1. valid_boxes: List[[xmin, ymin, xmax, ymax]]
        2. words: List[str]
        """
        h_orig, w_orig = image_bgr.shape[:2]

        # B1: Detect khung chữ
        horiz_boxes, _ = self.detector.detect(image_bgr)
        if not horiz_boxes or len(horiz_boxes[0]) == 0:
            return [], []

        raw_boxes = []
        for b in horiz_boxes[0]:
            x_min, x_max, y_min, y_max = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            raw_boxes.append([
                max(0, x_min),
                max(0, y_min),
                min(w_orig, x_max),
                min(h_orig, y_max)
            ])

        # B2: Sắp xếp thứ tự đọc tự nhiên
        sorted_boxes = self.sort_reading_order_boxes(raw_boxes)

        # B3: Cắt crop và nhận diện mặt chữ với VietOCR
        valid_boxes, words = [], []
        for box in sorted_boxes:
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_pil.crop((x1, y1, x2, y2))
            text = self.recognizer.predict(crop).strip()
            if text:
                valid_boxes.append(box)
                words.append(text)

        return valid_boxes, words