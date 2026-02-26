import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def ingest_data():
    print("🚀 Đang khởi động quy trình nạp dữ liệu tối ưu...")

    # 1. Load tài liệu
    loader_docx = DirectoryLoader("data", glob="**/*.docx", loader_cls=Docx2txtLoader)
    loader_pdf = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader_docx.load() + loader_pdf.load()

    # 2. Chia nhỏ văn bản (Sử dụng thông số tối ưu từ notebook)
    # Cắt theo thứ tự: Đoạn văn -> Dòng -> Câu -> Từ để giữ nguyên ý nghĩa pháp lý
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Đã tạo {len(chunks)} đoạn văn bản.")

    # 3. Tạo Vector Database (ChromaDB)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory="chroma_db_luat_new"
    )
    print("✅ Dữ liệu đã được lưu vào ChromaDB.")


if __name__ == "__main__":
    ingest_data()
