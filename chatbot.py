import os
import streamlit as st
import pandas as pd
from db_loader import get_all_jobs
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

@st.cache_data
def get_all_data():
    return get_all_jobs()

@st.cache_resource
def get_llm():
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY in environment variables.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        max_tokens=1024
    )
    return llm
