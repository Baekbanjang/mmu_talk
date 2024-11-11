# 메인
# mmu_talk_app.py

import streamlit as st
from modules.mmu_file_handler import DocumentProcessor
from modules.mmu_vector_store import VectorStoreManager
from modules.mmu_response_generator import ResponseGenerator
from modules.mmu_response_formatter import ResponseFormatter

def main():
    # 페이지 설정
    st.set_page_config(page_title="대화형 검색 시스템", layout="wide")  # Streamlit 페이지 설정
    
    # 세션 상태 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # 대화 내역 초기화
    
    st.title("목포해양대생을 위한 챗봇 - 뮤톡🐬")  # 페이지 제목
    st.subheader("🏫학교 생활에 대한 모든 것을 물어보세요🔎", divider='rainbow')  # 서브헤더 설정
    
    # 데이터 초기 처리
    if 'vector_store' not in st.session_state:
        with st.spinner("데이터를 처리하고 있습니다..."):  # 데이터 처리 중 스피너 표시
            # 처리된 파일 정보 표시
            text_files = DocumentProcessor.get_text_files()  # 텍스트 파일 목록 가져오기
            if text_files:
                st.info(f"발견된 데이터 파일: {', '.join(text_files.keys())}")  # 발견된 파일 정보 표시
                chunks = DocumentProcessor.process_multiple_text_files()  # 텍스트 파일 처리
                
                if chunks:
                    st.session_state.vector_store = VectorStoreManager.create_vector_store(chunks)  # 벡터 저장소 생성
                    st.success(f"총 {len(text_files)} 개의 파일 처리 완료!")  # 성공 메시지
                else:
                    st.error("데이터 처리에 실패했습니다.")  # 오류 메시지
                    st.stop()  # 실행 중단
            else:
                st.error(f"'data' 경로에서 텍스트 파일을 찾을 수 없습니다.")  # 파일 없음 메시지
                st.stop()  # 실행 중단

    # 사이드바에 대화 내역 지우기 버튼
    with st.sidebar:
        st.title("설정")  # 사이드바 제목
        if st.button('대화 내역 지우기'):  # 버튼 클릭 시
            st.session_state.chat_history = []  # 대화 내역 초기화
            st.rerun()  # 페이지 새로고침

    # 채팅 인터페이스
    if not st.session_state.chat_history:  # 대화 내역이 없는 경우
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "찾으시는 정보를 뮤톡🐬에게 남겨주세요!"  # 초기 안내 메시지
        })

    # 이전 대화 내용 표시
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):  # 사용자 또는 어시스턴트 메시지 표시
            st.markdown(ResponseFormatter.format_response(message["content"]))  # 포맷된 응답 표시

    # 새로운 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요"):  # 사용자 입력 받기
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": prompt})  # 대화 내역에 추가
        
        # 어시스턴트 응답 생성
        with st.spinner("답변을 생성하고 있습니다..."):  # 응답 생성 중 스피너 표시
            response, docs = ResponseGenerator.process_question(prompt, st.session_state.vector_store)  # 질문 처리
            formatted_response = ResponseFormatter.format_response(response)  # 응답 포맷팅
            
            # 응답을 대화 기록에 추가
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response  # 어시스턴트 응답 추가
            })
            
            # 페이지 새로고침
            st.rerun()  # 페이지 새로고침

if __name__ == "__main__":
    main()  # 메인 함수 실행