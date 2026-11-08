from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    xmin: int = Field(..., description="Tọa độ X góc trên bên trái")
    ymin: int = Field(..., description="Tọa độ Y góc trên bên trái")
    xmax: int = Field(..., description="Tọa độ X góc dưới bên phải")
    ymax: int = Field(..., description="Tọa độ Y góc dưới bên phải")


class DetectedBox(BaseModel):
    text: str = Field(..., description="Nội dung chữ")
    box: List[int] = Field(..., description="Tọa độ [xmin, ymin, xmax, ymax]")
    label: str = Field(..., description="Nhãn thực thể: MERCHANT_NAME, ADDRESS, DATE, TOTAL_AMOUNT, O")


class ReceiptResponse(BaseModel):
    status: str = Field(default="success", description="Trạng thái xử lý")
    merchant_name: Optional[str] = Field(None, description="Tên đơn vị bán lẻ / cửa hàng")
    address: Optional[str] = Field(None, description="Địa chỉ cửa hàng")
    date: Optional[str] = Field(None, description="Ngày giờ giao dịch")
    total_amount: Optional[int] = Field(None, description="Tổng số tiền thanh toán (VNĐ)")
    total_amount_raw: Optional[str] = Field(None, description="Chuỗi text tiền gốc trước khi làm sạch")
    category: Optional[str] = Field("Shopping", description="Phân loại chi tiêu tự động (Food, Shopping, Utilities,...)")
    image_width: Optional[int] = Field(None, description="Chiều rộng ảnh gốc (px)")
    image_height: Optional[int] = Field(None, description="Chiều cao ảnh gốc (px)")
    detected_boxes: List[DetectedBox] = Field(default_factory=list, description="Danh sách các box đã gán nhãn KIE")
    processing_time_ms: Optional[float] = Field(None, description="Thời gian xử lý của hệ thống (ms)")


class ExpenseCreateRequest(BaseModel):
    merchant_name: str
    address: Optional[str] = None
    date: Optional[str] = None
    total_amount: int
    category: str = "Shopping"


class ExpenseItem(BaseModel):
    id: int
    merchant_name: str
    address: Optional[str] = None
    date: Optional[str] = None
    total_amount: int
    category: str
    created_at: str


class FeedbackRequest(BaseModel):
    """Schema nhận phản hồi của người dùng khi AI đọc sai (Active Learning)"""
    image_name: str = Field(..., description="Tên file ảnh hóa đơn")
    corrected_merchant_name: Optional[str] = Field(None, description="Tên quán đúng")
    corrected_address: Optional[str] = Field(None, description="Địa chỉ đúng")
    corrected_date: Optional[str] = Field(None, description="Ngày giờ đúng")
    corrected_total_amount: Optional[int] = Field(None, description="Tổng tiền đúng")
