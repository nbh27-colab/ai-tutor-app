import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection
import fitz  # PyMuPDF

# --- Cấu hình ---
# Đường dẫn đến thư mục chứa dữ liệu học tập
DATA_PATH = "./data/kb/"
# Tên của Collection trong ChromaDB
COLLECTION_NAME = "ai_tutor_knowledge_base"

# --- Khởi tạo ChromaDB và Embedding Model ---
# Khởi tạo client ChromaDB cục bộ
client = chromadb.PersistentClient(path="./data/vector_db/chroma_db_store")
print("Khởi tạo ChromaDB client thành công.")

# Sử dụng một mô hình nhúng tiếng Việt
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="vinai/phobert-base-v2",
    device="cpu"  # Có thể đổi thành 'cuda' nếu bạn có GPU
)

# --- Các hàm hỗ trợ ---
def setup_collection() -> Collection:
    """Tạo hoặc lấy một collection ChromaDB."""
    try:
        # Xóa collection cũ nếu tồn tại để tạo mới
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Đã xóa collection cũ '{COLLECTION_NAME}'.")
    except Exception:
        pass  # Không làm gì nếu collection không tồn tại

    # Tạo collection mới với embedding function đã chọn
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    print(f"Đã tạo collection mới '{COLLECTION_NAME}'.")
    return collection

def load_documents_from_txt(filepath: str, documents: list, ids: list, filename: str):
    """Đọc và xử lý tài liệu từ tệp .txt."""
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        chunks = content.split("\n\n")
        for j, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append(chunk.strip())
                ids.append(f"{filename}_txt_{j}")

def load_documents_from_pdf(filepath: str, documents: list, ids: list, filename: str):
    """Đọc và xử lý tài liệu từ tệp .pdf."""
    try:
        doc = fitz.open(filepath)
        for j, page in enumerate(doc):
            content = page.get_text()
            # Chia nội dung trang thành các đoạn nhỏ
            chunks = content.split("\n")
            for k, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append(chunk.strip())
                    ids.append(f"{filename}_pdf_{j}_{k}")
        doc.close()
    except Exception as e:
        print(f"Lỗi khi xử lý tệp PDF {filepath}: {e}")

def load_documents(path: str) -> tuple[list, list]:
    """Đọc và xử lý tài liệu từ thư mục, hỗ trợ cả .txt và .pdf."""
    documents = []
    ids = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if filename.endswith(".txt"):
            load_documents_from_txt(filepath, documents, ids, filename)
        elif filename.endswith(".pdf"):
            load_documents_from_pdf(filepath, documents, ids, filename)
    print(f"Đã đọc và xử lý {len(documents)} đoạn văn bản từ các tệp.")
    return documents, ids

def main():
    """Chức năng chính để thiết lập cơ sở dữ liệu."""
    # 1. Đảm bảo thư mục dữ liệu tồn tại
    if not os.path.exists(DATA_PATH):
        print(f"Thư mục dữ liệu '{DATA_PATH}' không tồn tại. Vui lòng tạo thư mục và thêm tài liệu.")
        return

    # 2. Tải tài liệu
    documents, ids = load_documents(DATA_PATH)
    if not documents:
        print("Không tìm thấy tài liệu để xử lý.")
        return

    # 3. Thiết lập ChromaDB Collection
    collection = setup_collection()

    # 4. Thêm các tài liệu vào collection
    print("Đang thêm các tài liệu vào ChromaDB. Vui lòng đợi...")
    try:
        collection.add(
            documents=documents,
            ids=ids
        )
        print("Thêm tài liệu thành công!")
        print(f"Tổng số tài liệu trong collection '{collection.name}': {collection.count()}")
    except Exception as e:
        print(f"Lỗi khi thêm tài liệu vào ChromaDB: {e}")

if __name__ == "__main__":
    main()
