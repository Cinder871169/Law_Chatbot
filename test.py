import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

DB_FAISS_PATH = "vectorstore_luat"

# TẢI TÀI LIỆU VÀ TẠO VECTOR DATABASE
if os.path.exists(DB_FAISS_PATH):
    print("Đang tải Vector Database")
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
else:
    print("Đang quét")

    # DOCX
    loader_docx = DirectoryLoader("data", glob="**/*.docx", loader_cls=Docx2txtLoader)
    docs_docx = loader_docx.load()

    # PDF
    loader_pdf = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs_pdf = loader_pdf.load()

    docs = docs_docx + docs_pdf

    if len(docs) == 0:
        print("\nKhông tìm thấy file!")
        exit()

    print(f"-> Đã quét được tổng cộng: {len(docs)} tài liệu.")

    print("Đang cắt nhỏ tài liệu và embedding dữ liệu...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    print(f"-> Đã cắt tài liệu thành: {len(splits)} đoạn nhỏ.")

    if len(splits) == 0:
        print("\n[LỖI]: Quá trình cắt văn bản thất bại!")
        exit()

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
    print(f"-> Đã lưu Database tổng hợp vào thư mục: {DB_FAISS_PATH}")

# Cấu hình số lượng đoạn luật lấy ra
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. CẤU HÌNH GEMINI LLM VÀ PROMPT
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

system_prompt = (
    "Bạn là một trợ lý pháp lý ảo chuyên nghiệp. "
    "Sử dụng các đoạn văn bản luật được cung cấp dưới đây để trả lời câu hỏi của người dùng. "
    "Nếu thông tin không có trong văn bản được cung cấp, hãy nói rõ: 'Tôi không tìm thấy thông tin này trong tài liệu luật hiện tại', tuyệt đối không được tự bịa ra thông tin. "
    "Luôn trích dẫn nguồn (tên tài liệu hoặc Điều/Khoản nếu có) trong câu trả lời.\n\n"
    "Ngữ cảnh luật pháp:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# 5. TẠO RAG CHAIN
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 6. CHẠY CHATBOT
print("\n--- Hệ thống tra cứu luật đã sẵn sàng ---")
while True:
    user_query = input("\nBạn muốn hỏi gì (gõ 'exit' để thoát): ")
    if user_query.lower() == "exit":
        break

    try:
        response = rag_chain.invoke(user_query)
        print("\n[Trợ lý Pháp lý]:")
        print(response)
    except Exception as e:
        print(f"\n[Lỗi hệ thống]: {e}")
