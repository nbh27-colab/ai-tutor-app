import streamlit as st
import requests
import json

# --- Cấu hình ---
st.set_page_config(
    page_title="AI Tutor",
    page_icon="🧠",
    layout="centered"
)

# URL của FastAPI backend (chạy trên localhost)
API_URL = "http://localhost:8000"

st.title("Trợ Giảng AI 🧠")
st.markdown("Chào mừng bạn đến với hệ thống trợ giảng AI. Hãy nhập câu hỏi hoặc tải lên hình ảnh bài tập để nhận được câu trả lời chi tiết và tài liệu liên quan.")

# --- Form chính để nhập câu hỏi ---
with st.form(key='my_form', clear_on_submit=False):
    st.markdown("##### Nhập câu hỏi của bạn tại đây:")
    query = st.text_area("Câu hỏi (dạng văn bản):", height=100, label_visibility="collapsed")
    
    st.markdown("##### Hoặc tải lên ảnh bài tập:")
    uploaded_image = st.file_uploader("Hình ảnh:", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    submit_button = st.form_submit_button(label="Gửi Câu Hỏi")

if submit_button:
    if query:
        with st.spinner('Đang xử lý câu hỏi...'):
            try:
                # Gửi yêu cầu POST tới FastAPI
                response = requests.post(f"{API_URL}/ask_text", json={"query": query})
                response.raise_for_status() # Bắt lỗi HTTP
                data = response.json()
                st.session_state.response_data = data
            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi kết nối hoặc xử lý: {e}")
                st.session_state.response_data = None
    
    elif uploaded_image:
        with st.spinner('Đang xử lý hình ảnh...'):
            try:
                # Gửi yêu cầu POST hình ảnh tới FastAPI
                files = {'file': uploaded_image.getvalue()}
                response = requests.post(f"{API_URL}/ask_image", files=files)
                response.raise_for_status() # Bắt lỗi HTTP
                data = response.json()
                st.session_state.response_data = data
            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi kết nối hoặc xử lý: {e}")
                st.session_state.response_data = None
    
    else:
        st.warning("Vui lòng nhập câu hỏi hoặc tải lên hình ảnh.")
        st.session_state.response_data = None

# --- Hiển thị kết quả ---
if 'response_data' in st.session_state and st.session_state.response_data:
    st.divider()
    st.success("🎉 Câu trả lời đã sẵn sàng!")
    
    st.markdown("### Đáp Án")
    st.write(st.session_state.response_data['answer'])

    st.markdown("### Tài liệu & Bài tập Tương Tự")
    if st.session_state.response_data['similar_materials']:
        for material in st.session_state.response_data['similar_materials']:
            st.info(material)
    else:
        st.write("Không tìm thấy tài liệu tương tự.")
