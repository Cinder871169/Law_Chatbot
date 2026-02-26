import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Trợ lý Pháp lý AI", page_icon="⚖️")
st.title("⚖️ Trợ lý tra cứu Pháp luật")

with st.sidebar:
    st.title("⚙️ Tùy chọn")
    if st.button("🗑️ Xóa tin nhắn trên màn hình"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Xin chào! Tôi có thể giúp bạn tra cứu thông tin pháp luật gì hôm nay?",
            }
        ]
        st.success("Đã dọn dẹp màn hình!")

# --- ĐỌC API KEY ---
load_dotenv(override=True)
DB_CHROMA_PATH = "chroma_db_luat"

# Kiểm tra Database
if not os.path.exists(DB_CHROMA_PATH):
    st.error("Chưa có dữ liệu luật!")
    st.stop()

# --- KHỞI TẠO MẠNG VÀ TÌM KIẾM ---
# Kết nối vào DB có sẵn
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- XÂY DỰNG CHUỖI RAG ---
system_prompt = (
    "Bạn là một Luật sư ảo chuyên nghiệp, tận tâm và chính xác. "
    "Nhiệm vụ của bạn là sử dụng các đoạn văn bản luật được cung cấp dưới đây để giải đáp thắc mắc của người dùng.\n\n"
    "QUY TẮC TRẢ LỜI:\n"
    "1. Tính chính xác: Chỉ trả lời dựa trên nội dung có trong 'Ngữ cảnh luật pháp'. "
    "Nếu thông tin không có, hãy lịch sự trả lời: 'Rất tiếc, tôi không tìm thấy quy định này trong các văn bản luật hiện có trong hệ thống.'\n"
    "2. Trích dẫn nguồn: Mỗi câu trả lời BẮT BUỘC phải kèm theo tên Luật, số Điều và số Khoản cụ thể (ví dụ: Theo Điều 8, Luật Hôn nhân và Gia đình 2014).\n"
    "3. Cấu trúc: Sử dụng gạch đầu dòng cho các danh sách điều kiện hoặc thủ tục để người dùng dễ theo dõi.\n"
    "4. Tuyệt đối không tự bịa ra các con số hoặc thời hạn pháp lý nếu không thấy trong văn bản.\n\n"
    "Ngữ cảnh luật pháp:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | qa_prompt
    | llm
    | StrOutputParser()
)

# --- HIỂN THỊ GIAO DIỆN CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi có thể giúp bạn tra cứu thông tin pháp luật gì hôm nay?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu luật..."):
            try:
                response = rag_chain.invoke(user_query)
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Lỗi truy xuất: {e}")
