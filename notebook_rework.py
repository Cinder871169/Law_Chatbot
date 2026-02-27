import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


# --- KHỞI TẠO HỆ THỐNG TRUY XUẤT HỖN HỢP ---
@st.cache_resource
def get_hybrid_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory="vector_db_luat", embedding_function=embeddings
    )

    # Vector Retriever (Ngữ nghĩa)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Keyword Retriever (BM25 - Tìm chính xác Điều/Khoản)
    all_data = vectorstore.get()
    from langchain_core.documents import Document

    docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(all_data["documents"], all_data["metadatas"])
    ]
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 3

    # Kết hợp Hybrid (50/50)
    return EnsembleRetriever(
        retrievers=[keyword_retriever, vector_retriever], weights=[0.5, 0.5]
    )


# --- CẤU HÌNH CHAIN XỬ LÝ (LCEL) ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
retriever = get_hybrid_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn là Luật sư ảo chuyên nghiệp. Dùng ngữ cảnh sau để trả lời: {context}. Trích dẫn Điều/Khoản rõ ràng.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# LCEL Pipeline thay thế cho ConversationalRetrievalChain cũ
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["input"]))
    )
    | prompt
    | llm
    | StrOutputParser()
)

# --- GIAO DIỆN VÀ BỘ NHỚ (WINDOW MEMORY) ---
st.title("⚖️ Trợ lý Pháp luật Hybrid RAG")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Giới hạn 5 lượt chat gần nhất

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

if user_query := st.chat_input("Hỏi về pháp luật..."):
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        # Thực thi với 10 tin nhắn gần nhất trong bộ nhớ
        response = rag_chain.invoke(
            {"input": user_query, "chat_history": st.session_state.chat_history[-10:]}
        )
        st.write(response)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
