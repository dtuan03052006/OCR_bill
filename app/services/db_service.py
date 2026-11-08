import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent.parent / "expenses.db"


class DBService:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Khởi tạo bảng expenses nếu chưa tồn tại"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    merchant_name TEXT NOT NULL,
                    address TEXT,
                    date TEXT,
                    total_amount INTEGER NOT NULL,
                    category TEXT DEFAULT 'Shopping',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_expense(self, merchant_name: str, address: Optional[str], date: Optional[str], total_amount: int, category: str = "Shopping") -> int:
        """Lưu một khoản chi tiêu mới vào SQLite"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO expenses (merchant_name, address, date, total_amount, category)
                VALUES (?, ?, ?, ?, ?)
            """, (merchant_name, address, date, total_amount, category))
            conn.commit()
            return cursor.lastrowid

    def get_all_expenses(self) -> List[Dict[str, Any]]:
        """Lấy danh sách tất cả các khoản chi tiêu (sắp xếp mới nhất lên đầu)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, merchant_name, address, date, total_amount, category, created_at
                FROM expenses
                ORDER BY id DESC
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_expense(self, expense_id: int) -> bool:
        """Xóa khoản chi tiêu theo ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
            conn.commit()
            return cursor.rowcount > 0


# Singleton instance
db_service = DBService()
