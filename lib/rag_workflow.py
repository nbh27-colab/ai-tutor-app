import fitz
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from chromadb.utils import embedding_functions
import chromadb
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# --- Cấu hình và Khởi tạo ---
# Tên của Collection trong ChromaDB
COLLECTION_NAME = "ai_tutor_knowledge_base"
# Đường dẫn đến thư mục chứa ChromaDB
CHROMA_DB_PATH = "./data/vector_db/chroma_db_store"

# Khởi tạo các thành phần cốt lõi một lần duy nhất
# ChromaDB Client
db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
# Embedding Model (PhoBERT)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="vinai/phobert-base-v2",
    device="cpu"
)

# --- Định nghĩa Trạng thái (State) ---
# Dùng TypedDict để định nghĩa trạng thái của đồ thị LangGraph
class AgentState(TypedDict):
    """
    Biểu diễn trạng thái của tác nhân trong đồ thị.
    Mỗi node sẽ cập nhật trạng thái này.
    """
    query: str
    input_type: str
    context: List[str]
    answer: str
    similar_materials: List[str]
    image_path: str
    num_retrieved_docs: int

# --- Lớp LangGraph Workflow ---
class RagWorkflow:
    def __init__(self):
        # Khởi tạo ChromaDB Collection
        self.collection = db_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        # Khởi tạo mô hình LLM của OpenAI
        # Lấy API key từ biến môi trường
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Vui lòng đặt biến môi trường 'OPENAI_API_KEY'")

        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o-mini", # Có thể thay đổi model tùy theo nhu cầu
            temperature=0.7
        )

        # Xây dựng các node của LangGraph
        self.graph = StateGraph(AgentState)
        self.graph.add_node("ocr_node", self.ocr_processor)
        self.graph.add_node("retrieval_node", self.retrieval_processor)
        self.graph.add_node("generation_node", self.generation_processor)
        self.graph.add_node("suggestion_node", self.suggestion_processor)

        # Định nghĩa các cạnh và luồng
        self.graph.add_conditional_edges(
            START,
            self.route_input,
            {"image": "ocr_node", "text": "retrieval_node"}
        )
        self.graph.add_edge("ocr_node", "retrieval_node")
        self.graph.add_edge("retrieval_node", "generation_node")
        self.graph.add_edge("generation_node", "suggestion_node")
        self.graph.add_edge("suggestion_node", END)

        self.app = self.graph.compile()

    def route_input(self, state):
        """Định tuyến luồng dựa trên loại đầu vào (image/text)."""
        if state['input_type'] == 'image':
            return 'image'
        return 'text'

    def ocr_processor(self, state: AgentState) -> dict:
        """
        [Node] Xử lý OCR để chuyển đổi hình ảnh thành văn bản.
        Hiện tại là một placeholder, sẽ được hoàn thiện sau.
        """
        print("Đang xử lý OCR...")
        image_path = state.get('image_path')
        if not image_path or not Path(image_path).exists():
            print("Lỗi: Không tìm thấy file ảnh.")
            return {"query": ""}
        
        # TODO: Cần một mô hình OCR thực tế cho tiếng Việt tại đây
        # Đây là ví dụ đơn giản sử dụng PyMuPDF để trích xuất text từ PDF (giả định ảnh là 1 trang PDF)
        try:
            doc = fitz.open(image_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            print("Đã trích xuất văn bản từ ảnh.")
            return {"query": text}
        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return {"query": ""}


    def retrieval_processor(self, state: AgentState) -> dict:
        """
        [Node] Tìm kiếm và truy xuất các đoạn văn bản liên quan từ ChromaDB.
        """
        print("Đang truy xuất tài liệu...")
        query = state['query']
        # Tìm kiếm 5 tài liệu liên quan nhất
        results = self.collection.query(
            query_texts=[query],
            n_results=5
        )
        # Lấy nội dung của các tài liệu tìm được
        retrieved_docs = results.get('documents')[0] if results.get('documents') else []
        print(f"Đã truy xuất {len(retrieved_docs)} tài liệu.")
        return {"context": retrieved_docs, "num_retrieved_docs": len(retrieved_docs)}

    def generation_processor(self, state: AgentState) -> dict:
        """
        [Node] Tạo câu trả lời chi tiết bằng mô hình LLM.
        """
        print("Đang tạo câu trả lời...")
        query = state['query']
        context = "\n".join(state['context'])
        
        # Tạo prompt và gọi LLM
        if not context:
            answer = f"Xin lỗi, tôi không tìm thấy thông tin liên quan trong kho dữ liệu của bạn cho câu hỏi: '{query}'."
        else:
            prompt_template = PromptTemplate.from_template(
                "Dựa trên ngữ cảnh sau đây:\n---\n{context}\n---\n\n"
                "Hãy trả lời câu hỏi sau một cách chi tiết và dễ hiểu. "
                "Nếu câu hỏi yêu cầu giải một bài toán, hãy giải thích từng bước.\n"
                "Câu hỏi: {query}"
            )
            prompt = prompt_template.invoke({"context": context, "query": query})
            answer = self.llm.invoke(prompt).content

        return {"answer": answer}

    def suggestion_processor(self, state: AgentState) -> dict:
        """
        [Node] Đề xuất các tài liệu/bài tập tương tự.
        """
        print("Đang đề xuất tài liệu tương tự...")
        query = state['query']
        # Tìm kiếm thêm các tài liệu tương tự (ví dụ: lấy 3 tài liệu tiếp theo)
        results = self.collection.query(
            query_texts=[query],
            n_results=state['num_retrieved_docs'] + 3
        )
        
        similar_materials = []
        if results.get('documents'):
            # Loại bỏ các tài liệu đã dùng làm ngữ cảnh
            all_docs = results.get('documents')[0]
            context_docs = state['context']
            for doc in all_docs:
                if doc not in context_docs:
                    similar_materials.append(doc)
            
        return {"similar_materials": similar_materials}

    def run_workflow(self, query: str = "", image_path: str = "", input_type: str = "text") -> dict:
        """Chạy luồng xử lý chính."""
        initial_state = {
            "query": query,
            "input_type": input_type,
            "image_path": image_path,
            "context": [],
            "answer": "",
            "similar_materials": [],
            "num_retrieved_docs": 0,
        }
        final_state = self.app.invoke(initial_state)
        return final_state
