import os
import sys
import streamlit as st
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

# --- 1. CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(page_title="Trợ lý Pháp lý AI", page_icon="⚖️")
st.title("⚖️ Trợ lý tra cứu Pháp luật Việt Nam")
st.markdown("Hỏi tôi bất cứ điều gì về các bộ luật đã được cung cấp!")

# --- 2. CÀI ĐẶT API KEY ---
load_dotenv()


# --- 3. KHỞI TẠO HỆ THỐNG RAG (CHỈ CHẠY 1 LẦN) ---
@st.cache_resource(
    show_spinner="Đang khởi động hệ thống trí tuệ nhân tạo và đọc luật..."
)
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    DB_FAISS_PATH = "vectorstore_luat"

    if os.path.exists(DB_FAISS_PATH):
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        st.warning("Đang quét thư mục 'data' để tạo Database mới. Vui lòng đợi...")
        loader_docx = DirectoryLoader(
            "data", glob="**/*.docx", loader_cls=Docx2txtLoader
        )
        loader_pdf = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs = loader_docx.load() + loader_pdf.load()

        if len(docs) == 0:
            st.error("Lỗi: Không tìm thấy file dữ liệu nào trong thư mục 'data'.")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(DB_FAISS_PATH)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    system_prompt = (
        "Bạn là một trợ lý pháp lý ảo chuyên nghiệp. "
        "Sử dụng các đoạn văn bản luật được cung cấp dưới đây để trả lời câu hỏi của người dùng. "
        "Nếu thông tin không có trong văn bản được cung cấp, hãy nói rõ: 'Tôi không tìm thấy thông tin này trong tài liệu luật hiện tại'. "
        "Luôn trích dẫn nguồn (tên tài liệu hoặc Điều/Khoản nếu có) trong câu trả lời.\n\n"
        "Ngữ cảnh luật pháp:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# Gọi hàm khởi tạo
rag_chain = load_rag_chain()

# --- 4. XỬ LÝ LỊCH SỬ TRÒ CHUYỆN TRÊN GIAO DIỆN ---
# Khởi tạo bộ nhớ tạm để lưu tin nhắn trên giao diện
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi có thể giúp bạn tra cứu thông tin pháp luật gì hôm nay?",
        }
    ]

# Hiển thị lại các tin nhắn cũ
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 5. KHUNG NHẬP CHAT ---
if user_query := st.chat_input("Nhập câu hỏi pháp lý của bạn vào đây..."):
    # 1. Hiển thị câu hỏi của người dùng
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # 2. Xử lý câu trả lời của AI
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu luật..."):
            try:
                response = rag_chain.invoke(user_query)
                st.write(response)
                # Lưu câu trả lời vào lịch sử
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Lỗi hệ thống: {e}")
