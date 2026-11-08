from pathlib import Path
import torch
from pydantic_settings import BaseSettings
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # 1. App Info
    APP_NAME: str = "Vietnamese Receipt VDU System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # 2. Compute Device
    DEVICE: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Thiết bị tính toán: cuda hoặc cpu"
    )

    # 3. Model Paths
    VIETOCR_CONFIG: str = str(BASE_DIR / "weights" / "vietocr_configs" / "vgg_transformer.yml")
    VIETOCR_WEIGHTS: str = str(BASE_DIR / "weights" / "transformerocr.pth")
    LAYOUTLM_ONNX: str = str(BASE_DIR / "onnx_models" / "layoutlmv3_int8.onnx")
    PROCESSOR_DIR: str = str(BASE_DIR / "weights" / "layoutlmv3_finetuned")

    # 4. Heuristic Thresholds
    ORIENTATION_THRESHOLD: float = 1.15

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Khởi tạo một singleton instance để toàn bộ app dùng chung
settings = Settings()