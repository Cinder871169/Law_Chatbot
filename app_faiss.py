import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Trợ lý Pháp lý AI", page_icon="⚖️")
st.title("⚖️ Trợ lý Pháp luật")

# --- 2. CÀI ĐẶT API KEY ---
load_dotenv()
DB_FAISS_PATH = "faiss_db_luat"


# --- 3. NẠP HỆ THỐNG RAG ---
@st.cache_resource(show_spinner="Đang nạp dữ liệu pháp luật...")
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"Không tìm thấy Database tại {DB_FAISS_PATH}.")
        st.stop()

    # Nạp FAISS từ ổ cứng
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# Gọi hàm khởi tạo
rag_chain = load_rag_chain()

# --- 4. GIAO DIỆN CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi đã sẵn sàng hỗ trợ tra cứu luật cho bạn.",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("Nhập câu hỏi pháp lý..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            try:
                response = rag_chain.invoke(user_query)
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Lỗi: {e}")
