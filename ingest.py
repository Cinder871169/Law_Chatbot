import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def ingest_data():
    print("Đang quét dữ liệu từ thư mục 'data'...")

    # 1. Nạp song song cả PDF và DOCX với DirectoryLoader
    loader_pdf = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
    loader_docx = DirectoryLoader("data", glob="**/*.docx", loader_cls=Docx2txtLoader)

    documents = loader_pdf.load() + loader_docx.load()

    # 2. Chia nhỏ văn bản
    # Cắt theo thứ tự: Đoạn văn -> Dòng -> Câu -> Từ để giữ ngữ cảnh pháp lý
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Tạo Vector Database (ChromaDB) với Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory="vector_db_luat"
    )
    print(f"Thành công! Đã nạp {len(chunks)} đoạn văn bản từ PDF/Word vào Database.")


if __name__ == "__main__":
    ingest_data()
