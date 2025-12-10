# 🛒 RAG-powered Shop Assistant Chatbot

A smart product assistant chatbot built with **Retrieval-Augmented Generation (RAG)** architecture using **LangChain**, **Pinecone**, **Google Gemini**, and **Streamlit**.

---

## ✨ Features

✅ Tích hợp với cơ sở dữ liệu **SQL Server**  
✅ Sử dụng **Pinecone** cho tìm kiếm ngữ nghĩa  
✅ Tích hợp mô hình **Gemini 2.0 Flash** qua `langchain-google-genai`  
✅ Giao diện đơn giản, dễ dùng với **Streamlit**

---

## 🔧 Công nghệ sử dụng

| Công nghệ | Mô tả |
|----------|-------|
| 🧠 **LangChain** | Xử lý logic chuỗi và truy xuất ngữ cảnh |
| 🧬 **Pinecone** | Vector DB lưu trữ mô tả sản phẩm |
| 💬 **Google Gemini 2.0 Flash** | Tạo phản hồi từ ngữ cảnh truy vấn |
| 🗃️ **SQL Server** | Lưu trữ dữ liệu gốc: sản phẩm, biến thể... |
| 🖥️ **Streamlit** | Xây dựng giao diện người dùng |

---

## 🧠 Cách hoạt động
Người dùng đặt câu hỏi (ví dụ: "Điện thoại màu black nào rẻ nhất?")

Câu hỏi được chuyển thành embedding và truy vấn Pinecone

Các mô tả liên quan được trả về và kết hợp với câu hỏi

Gemini 2.0 Flash tạo phản hồi dựa trên truy vấn + ngữ cảnh

Trả lại kết quả cho người dùng qua giao diện

---

## 📌 Tài liệu tham khảo
LangChain Docs

Google AI Gemini API

Pinecone Docs

Streamlit Docs


