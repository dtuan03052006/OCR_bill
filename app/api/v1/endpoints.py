from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.schemas.receipt import ReceiptResponse, ExpenseCreateRequest, ExpenseItem
from app.pipeline.receipt_flow import ReceiptPipeline
from app.services.db_service import db_service

router = APIRouter(prefix="/v1", tags=["Receipt VDU & Expense Tracker"])
pipeline = ReceiptPipeline()


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "Receipt VDU & Expense Tracker Engine"}


@router.post(
    "/extract-receipt",
    response_model=ReceiptResponse,
    status_code=status.HTTP_200_OK,
    summary="Bóc tách thông tin hóa đơn tiếng Việt"
)
async def extract_receipt(file: UploadFile = File(...)):
    # 1. Kiểm tra định dạng ảnh
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File tải lên bắt buộc phải là định dạng hình ảnh (.jpg, .png, .jpeg)"
        )

    # 2. Đọc byte và đẩy vào pipeline
    try:
        contents = await file.read()
        result = pipeline.process(contents)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình xử lý ảnh: {str(e)}"
        )


@router.get(
    "/expenses",
    response_model=List[ExpenseItem],
    status_code=status.HTTP_200_OK,
    summary="Lấy danh sách các khoản chi tiêu đã lưu"
)
def get_expenses():
    return db_service.get_all_expenses()


@router.post(
    "/expenses",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Lưu hóa đơn vào Expense Tracker (SQLite)"
)
def save_expense(expense: ExpenseCreateRequest):
    new_id = db_service.add_expense(
        merchant_name=expense.merchant_name,
        address=expense.address,
        date=expense.date,
        total_amount=expense.total_amount,
        category=expense.category
    )
    return {"status": "success", "id": new_id, "message": "Đã lưu khoản chi thành công"}


@router.delete(
    "/expenses/{expense_id}",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Xóa một khoản chi tiêu"
)
def delete_expense(expense_id: int):
    success = db_service.delete_expense(expense_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy ID chi tiêu")
    return {"status": "success", "message": f"Đã xóa khoản chi #{expense_id}"}
