import os
import json
import glob
import streamlit as st
from dotenv import load_dotenv

# Import các thành phần RAG và Chat
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# --- 1. CẤU HÌNH LƯU TRỮ ---
HISTORY_DIR = "chat_histories"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)


def save_chat(title, messages):
    safe_title = "".join([c for c in title if c.isalnum() or c in (" ", "_")]).rstrip()
    file_path = os.path.join(HISTORY_DIR, f"{safe_title}.json")
    data = [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in messages
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_chat(title):
    file_path = os.path.join(HISTORY_DIR, f"{title}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [
                (
                    HumanMessage(content=m["content"])
                    if m["role"] == "user"
                    else AIMessage(content=m["content"])
                )
                for m in data
            ]
    return []


def delete_chat(title):
    file_path = os.path.join(HISTORY_DIR, f"{title}.json")
    if os.path.exists(file_path):
        os.remove(file_path)


# --- 2. HỆ THỐNG RAG HYBRID ---
@st.cache_resource
def init_rag():
    loader_pdf = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
    loader_docx = DirectoryLoader("data", glob="**/*.docx", loader_cls=Docx2txtLoader)
    documents = loader_pdf.load() + loader_docx.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 3

    return EnsembleRetriever(
        retrievers=[keyword_retriever, vector_retriever], weights=[0.5, 0.5]
    )


# --- 3. GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="Luật sư ảo Gemini", layout="wide")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat_title" not in st.session_state:
    st.session_state.current_chat_title = None

# Sidebar quản lý danh sách chat
with st.sidebar:
    st.title("📜 Lịch sử chủ đề")
    if st.button("➕ Đoạn chat mới", use_container_width=True):
        st.session_state.current_chat_title = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    existing_chats = [
        os.path.basename(f).replace(".json", "")
        for f in glob.glob(f"{HISTORY_DIR}/*.json")
    ]

    for title in sorted(existing_chats, reverse=True):
        cols = st.columns([0.8, 0.2])
        # Nút chọn chat
        if cols[0].button(f"📄 {title}", key=f"sel_{title}", use_container_width=True):
            st.session_state.current_chat_title = title
            st.session_state.messages = load_chat(title)
            st.rerun()
        # Nút xóa chat
        if cols[1].button("🗑️", key=f"del_{title}"):
            delete_chat(title)
            if st.session_state.current_chat_title == title:
                st.session_state.current_chat_title = None
                st.session_state.messages = []
            st.rerun()

# --- 4. XỬ LÝ LOGIC CHAT ---
display_title = (
    st.session_state.current_chat_title
    if st.session_state.current_chat_title
    else "Cuộc hội thoại mới"
)
st.title(f"⚖️ {display_title}")

retriever = init_rag()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn là Luật sư ảo chuyên nghiệp. Dùng ngữ cảnh: {context}. Trích dẫn Điều/Khoản rõ ràng.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: "\n\n".join(
            d.page_content for d in retriever.invoke(x["input"])
        )
    )
    | prompt
    | llm
    | StrOutputParser()
)

# Hiển thị lịch sử tin nhắn
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Nhập liệu từ người dùng
if user_input := st.chat_input("Hỏi về pháp luật..."):
    # BƯỚC 1: Nếu chưa có tiêu đề, tạo tiêu đề trước
    if st.session_state.current_chat_title is None:
        with st.spinner("Đang khởi tạo chủ đề..."):
            title_gen_prompt = (
                f"Tóm tắt câu hỏi sau thành tiêu đề cực ngắn (dưới 5 từ): {user_input}"
            )
            new_title = llm.invoke(title_gen_prompt).content.strip().replace('"', "")
            st.session_state.current_chat_title = new_title

    # BƯỚC 2: Thêm tin nhắn người dùng vào bộ nhớ và hiển thị
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # BƯỚC 3: Trả lời câu hỏi ngay lập tức
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu luật..."):
            response = rag_chain.invoke(
                {
                    "input": user_input,
                    "chat_history": st.session_state.messages[:-1][
                        -10:
                    ],  # Lấy lịch sử trước đó
                }
            )
            st.write(response)
            st.session_state.messages.append(AIMessage(content=response))

            # BƯỚC 4: Lưu vào file
            save_chat(st.session_state.current_chat_title, st.session_state.messages)

            # Rerun để tiêu đề trên cùng cập nhật theo chủ đề mới tạo
            st.rerun()
