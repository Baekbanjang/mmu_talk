# 문서 처리
# mmu_file_handler.py

import os
import streamlit as st
import re
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.mmu_config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    @staticmethod
    def get_text_files():
        text_files = {}
        try:
            for file in os.listdir(DATA_DIR):
                if file.endswith('.txt'):
                    category = os.path.splitext(file)[0]
                    text_files[category] = file
            return text_files
        except Exception as e:
            st.error(f"파일 목록 불러오기 실패: {e}")
            return {}

    @staticmethod
    def process_multiple_text_files():
        """텍스트 파일들을 처리하여 Document 객체 리스트 생성"""
        all_documents = []
        text_files = DocumentProcessor.get_text_files()
        
        try:
            if not text_files:
                raise Exception("텍스트 파일을 찾을 수 없습니다.")
                
            for category, filename in text_files.items():
                file_path = os.path.join(DATA_DIR, filename)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    
                    # URL 추출 (텍스트에서 URL 패턴 찾기)
                    url_pattern = r'https?://[^\s]+'
                    urls = re.findall(url_pattern, text)

                    sections = [section.strip() for section in text.split('\n\n') if section.strip()]
                    
                    for i, section in enumerate(sections):
                        lines = section.split('\n')
                        title = lines[0] if lines else ""
                        content = '\n'.join(lines[1:]) if len(lines) > 1 else section
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': file_path,
                                'category': category,
                                'section': i + 1,
                                'title': title,
                                'urls': urls  # URL 정보 추가
                            }
                        )
                        all_documents.append(doc)
            
            if not all_documents:
                raise Exception("처리할 문서가 없습니다.")
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=[
                    "\n\n",  # 큰 단락 구분
                    "\n1. ", "\n2. ", "\n3. ",  # 주요 항목 구분
                    "\n가. ", "\n나. ", "\n다. ", "\n라. ",  # 세부 항목 구분
                    "\n", # 일반 줄바꿈
                    ". ",  # 문장 구분
                    ", ", # 구문 구분
                    " "  # 단어 구분
                ],
                length_function=len,
                is_separator_regex=False
            )
            
            return text_splitter.split_documents(all_documents)
            
        except Exception as e:
            st.error(f"텍스트 파일 처리 중 오류 발생: {e}")
            return None