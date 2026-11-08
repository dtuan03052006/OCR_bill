from email.mime import image

import cv2
import numpy as np
from PIL import Image
from app.config import settings


class ImageService:
    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        """Giải mã byte stream của file upload thành ảnh OpenCV BGR"""
        # Gợi ý: dùng np.frombuffer và cv2.imdecode
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể giải mã ảnh từ dữ liệu byte")
        return img
        

    @staticmethod
    def auto_orient(image_bgr: np.ndarray) -> np.ndarray:
        """Tự động xoay thẳng hóa đơn nếu bị chụp nằm ngang"""
        h, w = image_bgr.shape[:2]
        if w > h * settings.ORIENTATION_THRESHOLD:
            # Gợi ý: dùng cv2.rotate với cv2.ROTATE_90_CLOCKWISE
            image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        return image_bgr

    @staticmethod
    def bgr_to_rgb_pil(image_bgr: np.ndarray) -> Image.Image:
        """Chuyển đổi ảnh OpenCV BGR sang PIL Image RGB"""
        # Gợi ý: cv2.cvtColor sang COLOR_BGR2RGB rồi Image.fromarray
        img_color = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_color)