import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from tempfile import TemporaryDirectory
from lib.rag_workflow import RagWorkflow

# --- Cấu hình ---
app = FastAPI(
    title="AI Tutor API",
    description="API hỗ trợ học sinh học tập bằng AI RAG.",
    version="1.0.0"
)

# Khởi tạo luồng xử lý RAG một lần duy nhất khi ứng dụng khởi động
try:
    rag_workflow = RagWorkflow()
except ValueError as e:
    raise RuntimeError(f"Lỗi khởi tạo RagWorkflow: {e}")

# --- Pydantic Models ---
class TextQuery(BaseModel):
    query: str

class ResponseData(BaseModel):
    answer: str
    similar_materials: list[str]

# --- Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Chào mừng đến với AI Tutor API!"}

@app.post("/ask_text", response_model=ResponseData)
async def ask_text(request: TextQuery):
    """
    Xử lý câu hỏi dạng văn bản từ học sinh.
    """
    try:
        result = rag_workflow.run_workflow(
            query=request.query,
            input_type="text"
        )
        return ResponseData(
            answer=result['answer'],
            similar_materials=result['similar_materials']
        )
    except Exception as e:
        print(f"Lỗi khi xử lý câu hỏi văn bản: {e}")
        raise HTTPException(status_code=500, detail="Có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại.")

@app.post("/ask_image", response_model=ResponseData)
async def ask_image(file: UploadFile = File(...)):
    """
    Xử lý câu hỏi dạng hình ảnh từ học sinh.
    """
    # Sử dụng thư mục tạm thời để lưu file ảnh
    try:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / file.filename
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            result = rag_workflow.run_workflow(
                image_path=str(temp_path),
                input_type="image"
            )

            return ResponseData(
                answer=result['answer'],
                similar_materials=result['similar_materials']
            )
    except Exception as e:
        print(f"Lỗi khi xử lý hình ảnh: {e}")
        raise HTTPException(status_code=500, detail="Có lỗi xảy ra trong quá trình xử lý hình ảnh. Vui lòng thử lại.")
