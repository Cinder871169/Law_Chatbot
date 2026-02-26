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
from langchain_chroma import Chroma

load_dotenv(override=True)

DB_CHROMA_PATH = "chroma_db_luat"

print("BẮT ĐẦU QUÁ TRÌNH NẠP DỮ LIỆU LUẬT...")

# 1. Quét file
loader_docx = DirectoryLoader("data", glob="**/*.docx", loader_cls=Docx2txtLoader)
loader_pdf = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader_docx.load() + loader_pdf.load()

if len(docs) == 0:
    print("Không tìm thấy file.")
    sys.exit()

print(f"- Đã tìm thấy {len(docs)} tài liệu.")

# 2. Cắt nhỏ văn bản
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print(f"- Đã cắt thành {len(splits)} đoạn nhỏ. Đang nhúng vào ChromaDB...")

# 3. Tạo và lưu Database
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
Chroma.from_documents(
    documents=splits, embedding=embeddings, persist_directory=DB_CHROMA_PATH
)

print(f"HOÀN THÀNH! Dữ liệu đã được lưu an toàn vào thư mục: '{DB_CHROMA_PATH}'")
