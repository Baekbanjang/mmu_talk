import os
from dotenv import load_dotenv
import streamlit as st
import re
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents.base import Document 

# === 설정 ===
load_dotenv()  # 환경 변수 로드
GOOGLE_API_KEY = st.secrets["google_api_key"]  # Streamlit 비밀 관리에서 Google API 키 가져오기
genai.configure(api_key=GOOGLE_API_KEY)  # Google Generative AI 설정

# 데이터 디렉토리 설정
DATA_DIR = "data"  # 데이터 파일이 저장된 디렉토리

# 모델 설정
EMBEDDING_MODEL = "models/embedding-001"  # 임베딩 모델
CHAT_MODEL = "gemini-1.5-flash"  # 채팅 모델

# 청크 설정
CHUNK_SIZE = 800
CHUNK_OVERLAP = 300
TOP_K = 4

def get_text_files():
    """데이터 디렉토리에서 모든 .txt 파일 찾기"""
    text_files = {}
    try:
        for file in os.listdir(DATA_DIR):
            if file.endswith('.txt'):
                # 파일 이름에서 확장자 제거하여 카테고리로 사용
                category = os.path.splitext(file)[0]
                text_files[category] = file
        return text_files
    except Exception as e:
        st.error(f"파일 목록 불러오기 실패: {e}")
        return {}

# === RAG 체인 개선 ===
def get_enhanced_rag_chain():
    """개선된 RAG 체인 생성"""
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
    
    prompt = PromptTemplate.from_template(template)
    model = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    return prompt | model | StrOutputParser()

def process_multiple_text_files():
    """텍스트 파일들을 처리하여 Document 객체 리스트 생성"""
    all_documents = []
    text_files = get_text_files()
    
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
                    
                    # 메타데이터에 URL 포함
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
            
        # 개선된 텍스트 분할기 설정
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
        
        chunked_documents = text_splitter.split_documents(all_documents)
        return chunked_documents
        
    except Exception as e:
        st.error(f"텍스트 파일 처리 중 오류 발생: {e}")
        return None

def create_vector_store(documents):
    """벡터 저장소 생성"""
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

def process_question(question: str, vector_store):
    """사용자 질문 처리 및 응답 생성"""
    try:
        # 이전 컨텍스트 저장을 위한 세션 상태 초기화
        if 'last_department_info' not in st.session_state:
            st.session_state.last_department_info = None
            
        retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
        docs = retriever.invoke(question)
        
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
        
        chain = get_enhanced_rag_chain()
        response = chain.invoke({
            "question": question,
            "context": context
        })
        
        return response, docs
    except Exception as e:
        st.error(f"질문 처리 중 오류 발생: {e}")
        return "죄송합니다. 응답을 생성하는 중 문제가 발생했습니다.", None

def format_response(response: str) -> str:
    """응답 텍스트를 보기 좋게 포매팅"""
    formatted_sections = []
    
    # 응답을 주요 섹션으로 분리
    sections = response.split('\n')
    current_section = []
    
    for line in sections:
        line = line.strip()
        if not line:
            continue
            
        # 새로운 섹션 시작
        if any(line.startswith(marker) for marker in ['📌', '📋', '📚', '💡']):
            if current_section:
                formatted_sections.append('\n'.join(current_section))
                current_section = []
            current_section.append(line)
            if not line.endswith(':\n\n'):  # 섹션 제목이 아닌 경우에만 빈 줄 추가
                current_section.append('')  # 섹션 내용과 제목 사이 빈 줄 추가
            continue
        
        # 불릿 포인트 처리
        if line.startswith('•'):
            # 여러 불릿 포인트가 한 줄에 있는 경우 분리
            bullet_points = line.split('•')[1:]  # 첫 번째 빈 문자열 제거
            for point in bullet_points:
                if point.strip():
                    current_section.append(f"  • {point.strip()}")
                    current_section.append('')  # 각 불릿 포인트 뒤에 빈 줄 추가
            continue
            
        current_section.append(line)
    
    # 마지막 섹션 추가
    if current_section:
        formatted_sections.append('\n'.join(current_section))
    
    # 최종 텍스트 조합
    formatted_text = '\n\n'.join(formatted_sections)
    
    # 중복되는 빈 줄 제거 및 정리
    lines = formatted_text.split('\n')
    cleaned_lines = []
    prev_empty = False
    
    for line in lines:
        if line.strip():
            cleaned_lines.append(line)
            prev_empty = False
        elif not prev_empty:
            cleaned_lines.append(line)
            prev_empty = True
    
    return '\n'.join(cleaned_lines)

def main():
    # 페이지 설정
    st.set_page_config(page_title="대화형 검색 시스템", layout="wide")
    
    # 세션 상태 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.title("목포해양대생을 위한 챗봇 - 뮤톡🐬")
    st.subheader("🏫학교 생활에 대한 모든 것을 물어보세요🔎", divider='rainbow')
    
    # 데이터 초기 처리
    if 'vector_store' not in st.session_state:
        with st.spinner("데이터를 처리하고 있습니다..."):
            # 처리된 파일 정보 표시
            text_files = get_text_files()
            if text_files:
                st.info(f"발견된 데이터 파일: {', '.join(text_files.keys())}")
                
                chunks = process_multiple_text_files()
                if chunks:
                    st.session_state.vector_store = create_vector_store(chunks)
                    st.success(f"총 {len(text_files)} 개의 파일 처리 완료!")
                else:
                    st.error("데이터 처리에 실패했습니다.")
                    st.stop()
            else:
                st.error(f"'{DATA_DIR}' 경로에서 텍스트 파일을 찾을 수 없습니다.")
                st.stop()

    # 사이드바에 대화 내역 지우기 버튼
    with st.sidebar:
        st.title("설정")
        if st.button('대화 내역 지우기'):
            st.session_state.chat_history = []
            st.rerun()

    # 채팅 인터페이스
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "찾으시는 정보를 뮤톡🐬에게 남겨주세요!"
        })

    # 이전 대화 내용 표시
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(format_response(message["content"]))

    # 새로운 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # 어시스턴트 응답 생성
        with st.spinner("답변을 생성하고 있습니다..."):
            response, docs = process_question(prompt, st.session_state.vector_store)
            formatted_response = format_response(response)
            
            # 응답을 대화 기록에 추가
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # 페이지 새로고침
            st.rerun()

if __name__ == "__main__":
    main()