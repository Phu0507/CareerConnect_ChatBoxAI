import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from db_loader import get_all_jobs

load_dotenv()


def search_vector_with_filter(query, llm, filter_dict=None, top_k=20):
    """
    Hỏi kiểu semantic + QA trên dữ liệu job.
    filter_dict: filter metadata (vd: {"location": {"$eq": "Hà Nội"}})
    """
    from langchain.chains import RetrievalQA

    vectorstore = init_vector_store()

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": top_k,
            "filter": filter_dict or {}
        }
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa_chain.run(query)
    return response


def init_vector_store():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "job-index"

    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY in environment variables.")

    # SDK Pinecone mới
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Tạo index
    existing = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/msmarco-distilbert-base-v4"
    )

    # LangChain wrapper
    vectorstore = LangchainPinecone(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # ===== Build documents từ bảng job =====
    df = get_all_jobs()
    documents = []

    for _, row in df.iterrows():
        job_loc = row.get("job_location") or "Không rõ"

        content = (
            f"Việc làm: {row['title']}. "
            f"Công ty: {row.get('company_name') or 'Không rõ'}. "
            f"Địa điểm: {job_loc}. "
            f"Mức lương: {int(row['salary_min']):,} - {int(row['salary_max']):,} VNĐ. "
            f"Trình độ: {row.get('education_level')}. "
            f"Kinh nghiệm: {row.get('years_of_experience')}. "
            f"Mô tả: {row.get('description')}. "
            f"Quyền lợi: {row.get('benefits')}."
        )

        metadata = {
            "source": "SQL",
            "job_id": row["job_id"],
            "location": job_loc,
            "salary_min": row["salary_min"],
            "salary_max": row["salary_max"],
            "job_type": row["job_type"],
            "company_name": row.get("company_name"),
        }

        documents.append(Document(page_content=content, metadata=metadata))

    if documents:
        vectorstore.add_documents(documents)

    return vectorstore
