# 벡터 저장소 관리
# mmu_vector_store.py

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from modules.mmu_config import GOOGLE_API_KEY, EMBEDDING_MODEL, TOP_K

class VectorStoreManager:
    @staticmethod
    def create_vector_store(documents):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY
            )
            vector_store = FAISS.from_documents(documents, embedding=embeddings)
            vector_store.save_local("faiss_index")
            return vector_store
        except Exception as e:
            st.error(f"벡터 저장소 생성 중 오류 발생: {e}")
            return None