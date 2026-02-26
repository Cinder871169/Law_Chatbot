# ⚖️ Chatbot hỗ trợ tra cứu luật

Chatbot tra cứu văn bản pháp luật Việt Nam sử dụng công nghệ RAG (Retrieval-Augmented Generation). Hệ thống cho phép nạp các file luật định dạng PDF/Word và trả lời câu hỏi dựa trên nội dung thực tế, đảm bảo tính chính xác và có trích dẫn nguồn.

- Hỗ trợ nạp file `.pdf` và `.docx`.
- Tùy chọn sử dụng **ChromaDB** hoặc **FAISS**.
- AI chỉ trả lời dựa trên dữ liệu được cung cấp, không tự bịa quy định.
- Xây dựng trên nền tảng Streamlit, dễ dàng sử dụng.

- **Ngôn ngữ:** Python 3.11
- **LLM:** Google Gemini 2.5 Flash
- **Framework:** LangChain (LCEL)
- **Vector Database:** ChromaDB / FAISS
- **Giao diện:** Streamlit

## Cài đặt và sử dụng
1. Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate
2. Cài đặt thư viện
pip install -r requirements.txt
3. Thêm API Key
Tạo file .env và thêm Gemini API Key: GOOGLE_API_KEY='Dien key vao day'
4. Chuẩn bị dữ liệu và khởi chạy
  - Bước 1: Nạp luật vào Database (Chỉ cần chạy khi có file mới trong thư mục data/)
  python load_data_faiss.py (python load_data_chromadb.py) hoặc chạy trực tiếp
  - Bước 2: Khởi động giao diện Chatbot
  streamlit run app_faiss.py (streamlit run app_chromadb.py)

## 📦 Cấu trúc thư mục
```text
├── data/                 # Thư mục chứa các file luật (.pdf, .docx)
├── faiss_db_luat/        # Database FAISS (tự động tạo khi chạy load_data)
├── chroma_db_luat/       # Database ChromaDB (tự động tạo khi chạy load_data)
├── .env                  # Lưu API Key
├── load_data_faiss.py    # Xử lý file dữ liệu luật, embedding và nạp vào DB sử dụng FAISS
├── load_data_chromadb.py # Xử lý file dữ liệu luật, embedding và nạp vào DB sử dụng ChromaDB
├── app_chromadb.py       # File chạy giao diện Chat chính với dữ liệu ChromaDB đã nạp
├── app_faiss.py          # File chạy giao diện Chat chính với dữ liệu FAISS đã nạp
├── requirements.txt      # Danh sách thư viện cần cài đặt
├── vectorstore_luat      # Database thử nghiệm
├── test.py               # File chatbot thử nghiệm chạy bằng terminal.
├── app.py                # Ứng dụng chatbot thử nghiệm ban đầu
└── README.md             # Hướng dẫn sử dụng
