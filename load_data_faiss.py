import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DB_FAISS_PATH = "faiss_db_luat"

print("BẮT ĐẦU NẠP DỮ LIỆU LUẬT VÀO FAISS...")

# 1. Quét file từ thư mục data
print("- Đang quét thư mục 'data'...")
if not os.path.exists("data"):
    print("LỖI: Thư mục 'data' không tồn tại!")
    sys.exit()

loader_docx = DirectoryLoader("data", glob="**/*.docx", loader_cls=Docx2txtLoader)
loader_pdf = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader_docx.load() + loader_pdf.load()

if len(docs) == 0:
    print("LỖI: Không tìm thấy filenào!")
    sys.exit()

print(f"- Đã nạp {len(docs)} tài liệu.")

# 2. Chia nhỏ văn bản để AI dễ tra cứu
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(docs)
print(f"- Tổng cộng có {len(splits)} đoạn văn bản sau khi cắt.")

# 3. Chuyển đổi văn bản thành Vector và lưu xuống ổ cứng
print("- Đang tạo Vector Embeddings ...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
vectorstore.save_local(DB_FAISS_PATH)

print(f"HOÀN THÀNH! Database FAISS đã được lưu tại: '{DB_FAISS_PATH}'")
