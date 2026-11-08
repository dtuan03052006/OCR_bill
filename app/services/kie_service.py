import re
from typing import Dict, List, Optional
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

from app.config import settings


class KIEService:
    LABELS = [
        "O",
        "B-MERCHANT_NAME", "I-MERCHANT_NAME",
        "B-ADDRESS", "I-ADDRESS",
        "B-DATE", "I-DATE",
        "B-TOTAL_AMOUNT", "I-TOTAL_AMOUNT"
    ]
    id2label = {i: l for i, l in enumerate(LABELS)}

    def __init__(self):
        print(f"[*] Đang khởi tạo KIEService với ONNX: {settings.LAYOUTLM_ONNX}...")
        # 1. Khởi tạo Processor
        self.processor = AutoProcessor.from_pretrained(
            settings.PROCESSOR_DIR,
            apply_ocr=False
        )

        # 2. Cấu hình ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if settings.DEVICE == "cuda" else ["CPUExecutionProvider"]
        
        self.session = ort.InferenceSession(
            settings.LAYOUTLM_ONNX,
            sess_options,
            providers=providers
        )
        print("[+] Khởi tạo KIEService thành công!")

    @staticmethod
    def clean_repeated_words(text: str) -> str:
        """Khử lặp từ do OCR ('thế thế thế' -> 'thế')"""
        if not text:
            return ""
        words = text.split()
        cleaned = []
        for w in words:
            if not cleaned or w.lower() != cleaned[-1].lower():
                cleaned.append(w)
        return " ".join(cleaned).strip()

    @staticmethod
    def parse_vietnamese_amount(text: str) -> Optional[int]:
        """Chuẩn hóa số tiền Việt Nam và loại trừ số điện thoại/hotline"""
        if not text:
            return None
        clean = re.sub(r"[^\d.,]", "", text)
        if not clean:
            return None
        digits = re.sub(r"[^\d]", "", clean)
        if not digits:
            return None
        # Loại trừ số điện thoại 10 số (09x, 08x, 03x, 07x, 05x)
        if len(digits) == 10 and digits.startswith(("09", "08", "03", "07", "05")):
            return None
        val = int(digits)
        if 1000 <= val <= 100_000_000:
            return val
        return None

    def extract_total_amount_advanced(self, words: List[str], boxes: List[List[int]], ai_labels: Optional[Dict[int, str]] = None) -> Optional[int]:
        PRIMARY_ANCHORS = [
            "tổng cộng", "tong cong", "tổng tiền", "tong tien",
            "thanh toán", "thanh toan", "tổng thanh toán", "tong thanh toan",
            "cộng tiền", "cong tien", "cộng", "tiền hàng", "khách phải trả", "khach phai tra",
            "cần thanh toán", "can thanh toan", "tổng:", "total", "grand total"
        ]
        SECONDARY_ANCHORS = [
            "tổng", "tong", "amount", "phải trả"
        ]
        NEGATIVE_KEYWORDS = [
            "chiết khấu", "chiet khau", "giảm giá", "giam gia", "đã giảm", "da giam",
            "tổng tiền đã giảm", "tiền thừa", "tien thua", "tiền trả lại", "tien tra lai",
            "đơn giá", "don gia", "số lượng", "sl", "điểm"
        ]

        # Tầng 1: Trong cùng 1 box chứa cả Anchor và tiền
        for i, w in enumerate(words):
            w_lower = w.lower()
            for anchor in PRIMARY_ANCHORS:
                if anchor in w_lower:
                    sub_text = w_lower.split(anchor)[-1]
                    val = self.parse_vietnamese_amount(sub_text)
                    if val:
                        return val

        # Tầng 2: Chấm điểm không gian (Spatial Scoring)
        anchors = []
        for i, w in enumerate(words):
            w_lower = w.lower()
            if any(neg in w_lower for neg in NEGATIVE_KEYWORDS):
                continue
            for p in PRIMARY_ANCHORS:
                if p in w_lower:
                    anchors.append((i, 2.0))
                    break
            else:
                for s in SECONDARY_ANCHORS:
                    if s in w_lower:
                        anchors.append((i, 1.0))
                        break

        candidates = []
        max_y = max([box[3] for box in boxes] or [1])
        for i, (w, b) in enumerate(zip(words, boxes)):
            val = self.parse_vietnamese_amount(w)
            if val is None:
                continue

            is_negative = False
            if i > 0 and any(neg in words[i-1].lower() for neg in NEGATIVE_KEYWORDS):
                is_negative = True

            score = 0.0
            if ai_labels and ai_labels.get(i) == "TOTAL_AMOUNT":
                score += 35.0
            if is_negative:
                score -= 50.0

            b_h = max(b[3] - b[1], 15)
            b_cy = (b[1] + b[3]) / 2

            for a_idx, a_weight in anchors:
                a_box = boxes[a_idx]
                a_h = max(a_box[3] - a_box[1], 15)
                a_cy = (a_box[1] + a_box[3]) / 2

                if b[0] >= a_box[0] - 10 and abs(a_cy - b_cy) <= a_h * 1.3:
                    dx = b[0] - a_box[2]
                    score += (60.0 * a_weight) - (dx * 0.03)
                elif (0 < (b[1] - a_box[3]) <= a_h * 2.8) and (abs(b[0] - a_box[0]) < 300):
                    dy = b[1] - a_box[3]
                    score += (45.0 * a_weight) - (dy * 0.05)

            y_ratio = b[1] / max_y
            if y_ratio > 0.5:
                score += 15.0 * y_ratio

            candidates.append((score, val, b[1]))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            if candidates[0][0] > 0:
                return candidates[0][1]

        # Tầng 3: Fallback đáy hóa đơn
        bottom_candidates = [c for c in candidates if c[2] > 0.6 * max_y]
        if bottom_candidates:
            bottom_candidates.sort(key=lambda x: x[1], reverse=True)
            return bottom_candidates[0][1]

        return None

    def extract_entities(self, image_pil: Image.Image, words: List[str], valid_boxes: List[List[int]]) -> dict:
        if not words or not valid_boxes:
            return {}

        w_orig, h_orig = image_pil.size

        # 1. Chuẩn hóa Bounding Box sang [0, 1000]
        norm_boxes = [
            [
                max(0, min(1000, int(1000 * (b[0] / w_orig)))),
                max(0, min(1000, int(1000 * (b[1] / h_orig)))),
                max(0, min(1000, int(1000 * (b[2] / w_orig)))),
                max(0, min(1000, int(1000 * (b[3] / h_orig)))),
            ]
            for b in valid_boxes
        ]

        # 2. Xử lý ảnh và Tokenize
        image_inputs = self.processor.image_processor(images=image_pil, return_tensors="np")
        pixel_values = image_inputs["pixel_values"]

        tokenized = self.processor.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=512
        )

        word_ids = tokenized.word_ids()
        aligned_boxes = [
            norm_boxes[w_idx] if w_idx is not None else [0, 0, 0, 0]
            for w_idx in word_ids
        ]

        # 3. Chuẩn bị Inputs cho ONNX Runtime (numpy arrays int64 và float32)
        onnx_inputs = {
            "input_ids": np.array([tokenized["input_ids"]], dtype=np.int64),
            "bbox": np.array([aligned_boxes], dtype=np.int64),
            "attention_mask": np.array([tokenized["attention_mask"]], dtype=np.int64),
            "pixel_values": pixel_values.astype(np.float32)
        }

        # 4. ONNX Inference
        outputs = self.session.run(None, onnx_inputs)
        logits = outputs[0]  # shape [1, 512, num_labels]
        preds = np.argmax(logits, axis=2)[0]

        # 5. Ánh xạ về từng từ gốc (Sub-token pooling: Majority voting giữa các sub-tokens)
        from collections import defaultdict
        word_token_tags = defaultdict(list)
        for idx, w_idx in enumerate(word_ids):
            if w_idx is not None:
                tag = self.id2label.get(preds[idx], "O").replace("B-", "").replace("I-", "")
                word_token_tags[w_idx].append(tag)

        word_labels = {}
        for w_idx, tags in word_token_tags.items():
            non_o_tags = [t for t in tags if t != "O"]
            if non_o_tags:
                best_tag = max(set(non_o_tags), key=non_o_tags.count)
                word_labels[w_idx] = best_tag
            else:
                word_labels[w_idx] = "O"

        # 6. Ràng buộc vị trí không gian (Spatial Constraints)
        h_max = max([b[3] for b in valid_boxes] or [1])
        extracted = {"MERCHANT_NAME": [], "ADDRESS": [], "DATE": [], "TOTAL_AMOUNT": []}
        TABLE_HEADERS = [
            "tên hàng", "đơn giá", "thành tiền", "số lượng", "sl", "đvt",
            "giá bán", "tiền khách trả", "tiền trả lại", "giá đăng ký",
            "hóa đơn", "phiếu thanh toán", "bán hàng", "bán lẻ", "vat"
        ]

        detected_boxes = []
        for i, (box, word) in enumerate(zip(valid_boxes, words)):
            label = word_labels.get(i, "O")
            word_lower = word.lower()
            y_center = (box[1] + box[3]) / 2
            y_ratio = y_center / h_max

            if label == "MERCHANT_NAME":
                if y_ratio > 0.40 or any(h in word_lower for h in TABLE_HEADERS) or "hóa đơn" in word_lower:
                    label = "O"

            if label == "ADDRESS":
                if y_ratio > 0.50 or any(h in word_lower for h in TABLE_HEADERS) or "hóa đơn" in word_lower:
                    label = "O"

            if label == "TOTAL_AMOUNT":
                digits = re.sub(r"[^\d]", "", word)
                if y_ratio < 0.35 or not digits or len(digits) < 4:
                    label = "O"

            if label == "DATE":
                if not re.search(r"\d", word) or any(k in word_lower for k in ["sđt", "giá bán", "đt", "tel", "mã"]):
                    label = "O"

            if label != "O":
                extracted[label].append(word)
                detected_boxes.append({
                    "text": word,
                    "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    "label": label
                })

        # 7. Hậu xử lý kết quả
        date_candidates = " ".join(extracted["DATE"])
        date_match = re.search(
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:\s*-?\s*\d{1,2}:\d{2})?",
            date_candidates or " ".join(words)
        )

        # Trích xuất số tiền AI
        ai_amounts = [self.parse_vietnamese_amount(x) for x in extracted["TOTAL_AMOUNT"]]
        ai_amounts = [a for a in ai_amounts if a is not None]

        # Trích xuất số tiền Spatial Anchor (tìm từ khóa 'Tổng cộng', 'Tổng thanh toán', 'Cộng tiền'...)
        spatial_amount = self.extract_total_amount_advanced(words, valid_boxes, word_labels)

        # Quyết định tiền cuối cùng:
        if spatial_amount is not None and (not ai_amounts or spatial_amount in ai_amounts):
            final_total = spatial_amount
        elif ai_amounts:
            final_total = max(ai_amounts)
        else:
            final_total = spatial_amount

        # Phân loại danh mục chi tiêu tự động (Expense Category)
        clean_merchant = self.clean_repeated_words(" ".join(extracted["MERCHANT_NAME"]))
        merchant_lower = clean_merchant.lower()
        all_text_lower = " ".join(words).lower()

        if any(k in merchant_lower or k in all_text_lower for k in [
            "sữa chua", "cà phê", "cafe", "trà", "quán", "nhà hàng", "bún", "phở", "cơm",
            "bánh", "pizza", "lẩu", "mì", "food", "restaurant", "nước giải khát"
        ]):
            category = "Food"
        elif any(k in merchant_lower or k in all_text_lower for k in [
            "xăng", "dầu", "petrol", "tiền điện", "tiền nước", "internet", "cước", "viettel", "vinaphone"
        ]):
            category = "Utilities"
        elif any(k in merchant_lower or k in all_text_lower for k in [
            "thuốc", "pharmacy", "bệnh viện", "phòng khám", "y tế"
        ]):
            category = "Health"
        elif any(k in merchant_lower or k in all_text_lower for k in [
            "rạp", "cinema", "phim", "karaoke", "game", "giải trí"
        ]):
            category = "Entertainment"
        else:
            category = "Shopping"

        return {
            "merchant_name": clean_merchant,
            "address": self.clean_repeated_words(" ".join(extracted["ADDRESS"])),
            "date": date_match.group(0) if date_match else self.clean_repeated_words(date_candidates),
            "total_amount_raw": " ".join(extracted["TOTAL_AMOUNT"]).strip() or None,
            "total_amount": final_total,
            "category": category,
            "detected_boxes": detected_boxes
        }