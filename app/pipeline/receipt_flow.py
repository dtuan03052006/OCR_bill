import time
from app.schemas.receipt import ReceiptResponse
from app.services.image_service import ImageService
from app.services.ocr_service import OCRService
from app.services.kie_service import KIEService


class ReceiptPipeline:
    def __init__(self):
        self.image_service = ImageService()
        self.ocr_service = OCRService()
        self.kie_service = KIEService()
        print("[+] ReceiptPipeline đã sẵn sàng phục vụ!")

    def process(self, image_bytes: bytes) -> ReceiptResponse:
        t0 = time.perf_counter()

        # 1. Giải mã và làm thẳng ảnh
        image_bgr = self.image_service.decode_image(image_bytes)
        image_bgr = self.image_service.auto_orient(image_bgr)
        image_pil = self.image_service.bgr_to_rgb_pil(image_bgr)

        # 2. Text Detection & Recognition
        valid_boxes, words = self.ocr_service.extract_text(image_bgr, image_pil)
        if not words:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return ReceiptResponse(
                status="no_text_detected",
                processing_time_ms=round(elapsed_ms, 2)
            )

        # 3. KIE Extraction (ONNX + Heuristic)
        entities = self.kie_service.extract_entities(image_pil, words, valid_boxes)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 4. Đóng gói Pydantic Model
        return ReceiptResponse(
            status="success",
            merchant_name=entities.get("merchant_name") or None,
            address=entities.get("address") or None,
            date=entities.get("date") or None,
            total_amount=entities.get("total_amount"),
            total_amount_raw=entities.get("total_amount_raw"),
            category=entities.get("category", "Shopping"),
            image_width=image_pil.width,
            image_height=image_pil.height,
            detected_boxes=entities.get("detected_boxes", []),
            processing_time_ms=round(elapsed_ms, 2)
        )