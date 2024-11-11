# 응답 생성
# mmu_respones_generator.py
import streamlit as st
import google.generativeai as genai
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from modules.mmu_config import GOOGLE_API_KEY, CHAT_MODEL, TOP_K


class ResponseGenerator:
    @staticmethod
    # === RAG 체인 개선 ===
    def get_enhanced_rag_chain():
        """개선된 RAG 프롬프트 체인 생성"""
        template = """
        다음의 컨텍스트를 기반으로 질문에 답변해주세요:

        컨텍스트: {context}

        질문: {question}

        답변 작성 규칙:
        1. 핵심 내용을 먼저 2-3줄로 요약
        2. 상세 내용을 불릿 포인트로 구분
        3. 중요 내용은 **강조** 처리
        4. 참고 섹션 작성 규칙:
          - 담당부서가 있으면 "- 담당부서: [부서명](☎ xxx-xxxx)" 형식으로 첫 줄에 표시
           - URL이 있으면 "* 항목명: URL" 형식으로 표시
        5. 불릿 포인트는 붙여서 시작

        답변 형식:
        📌 핵심 요약:
        (간단한 요약 제공)

        📋 상세 내용:
        •**중요 내용 1**
        •**중요 내용 2**
        •상세 내용 3
        (각 항목을 새로운 줄에 시작)

        📚 참고:
        (담당부서 정보가 있는 경우 표시)
        (URL 정보가 있는 경우 표시)
        """

        prompt = PromptTemplate.from_template(template)  # 프롬프트 템플릿 생성
        model = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )  # 채팅 모델 초기화
        return prompt | model | StrOutputParser()  # 프롬프트, 모델, 출력 파서를 연결하여 반환
    
    @staticmethod
    def process_question(question: str, vector_store):
        """사용자 질문 처리 및 응답 생성"""
        try:
            # 이전 컨텍스트 저장을 위한 세션 상태 초기화
            if 'last_department_info' not in st.session_state:
               st.session_state.last_department_info = None

            # 관련 문서 검색
            retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})  # 벡터 저장소에서 검색기 생성
            docs = retriever.invoke(question)  # 질문에 대해 관련 문서 검색

            # URL 정보 수집
            urls = []
            department_info = None

            # 현재 문서에서 담당부서 정보 찾기
            for doc in docs:
                if 'urls' in doc.metadata and doc.metadata['urls']:
                    urls.extend(doc.metadata['urls'])
            
                # 담당부서 정보 찾기
                content = doc.page_content
                dept_pattern = r'담당부서[^\n]*?([^()]+\(☎\s*\d{3}-\d{4}\))'
                dept_matches = re.findall(dept_pattern, content, re.DOTALL)
            
                if dept_matches:
                    department_info = dept_matches[0].strip()
                    st.session_state.last_department_info = department_info  # 찾은 정보 저장
                    break

            # 현재 문서에서 찾지 못했다면 이전 정보 사용
            if not department_info and st.session_state.last_department_info:
                department_info = st.session_state.last_department_info
            
            urls = list(set(urls))  # 중복 제거

                # 컨텍스트 구성
            context = "\n\n".join(doc.page_content for doc in docs)
            if department_info:
                context += f"\n\nDEPARTMENT_INFO: {department_info}"
            if urls:
                context += "\n\nURL_LIST: " + "\n".join(urls)

            # RAG 체인으로 응답 생성
            chain = ResponseGenerator.get_enhanced_rag_chain()  # 개선된 RAG 체인 가져오기
            response = chain.invoke({
                "question": question,
                "context": context
            })
        
            return response, docs  # 응답과 문서 반환
        except Exception as e:
            st.error(f"질문 처리 중 오류 발생: {e}")  # 오류 발생 시 에러 메시지 출력
            return "죄송합니다. 응답을 생성하는 중 문제가 발생했습니다.", None
        


