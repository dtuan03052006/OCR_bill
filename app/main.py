from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.api.v1.endpoints import router as api_v1_router

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Hệ thống bóc tách thông tin hóa đơn tiếng Việt (VDU) sử dụng EasyOCR, VietOCR và LayoutLMv3 ONNX Runtime."
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gắn static files và trang chủ giao diện Camera
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", summary="Giao diện Web Chụp ảnh & Bóc tách Hóa đơn")
@app.get("/dashboard", include_in_schema=False)
@app.get("/invoices", include_in_schema=False)
@app.get("/invoices/{subpath:path}", include_in_schema=False)
@app.get("/analytics", include_in_schema=False)
@app.get("/settings", include_in_schema=False)
async def serve_camera_ui():
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)


# Gắn router API
app.include_router(api_v1_router, prefix="/api")


