import os
import sys
import streamlit as st
from dotenv import load_dotenv

# --- 1. CẤU HÌNH HỆ THỐNG ---
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(override=True)

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- 2. GIAO DIỆN ---
st.set_page_config(page_title="Chatbot hỗ trợ tra cứu luật", page_icon="⚖️")
st.title("⚖️ Chatbot Hỗ Trợ Tra Cứu Luật")


# --- 3. KHỞI TẠO RETRIEVER
@st.cache_resource
def get_hybrid_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Kiểm tra database
    if not os.path.exists("chroma_db_luat"):
        st.error("Không tìm thấy thư mục 'chroma_db_luat'.")
        st.stop()

    vectorstore = Chroma(
        persist_directory="chroma_db_luat", embedding_function=embeddings
    )

    # 3.1. Vector Search (Tìm theo ngữ nghĩa)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3.2. Keyword Search (BM25 - Tìm chính xác Điều/Khoản)
    all_data = vectorstore.get()
    from langchain_core.documents import Document

    docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(all_data["documents"], all_data["metadatas"])
    ]

    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 3

    # 3.3. Ensemble
    return EnsembleRetriever(
        retrievers=[keyword_retriever, vector_retriever], weights=[0.5, 0.5]
    )


# --- 4. XÂY DỰNG CHUỖI XỬ LÝ (LCEL CHAIN) ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
retriever = get_hybrid_retriever()

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn là Luật sư ảo chuyên nghiệp. Dùng ngữ cảnh dưới đây để trả lời câu hỏi. Trích dẫn Điều/Khoản rõ ràng.\n\nNgữ cảnh:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["input"]))
    )
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. QUẢN LÝ LỊCH SỬ CHAT (SESSION STATE) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Lưu trữ tối đa 5 lượt hội thoại

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

if user_query := st.chat_input("Hỏi về pháp luật..."):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # Thực thi với lịch sử chat (Window Memory k=5)
        response = rag_chain.invoke(
            {"input": user_query, "chat_history": st.session_state.chat_history[-10:]}
        )
        st.write(response)

        # Cập nhật bộ nhớ hội thoại
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
