# 설정 관리
# mmu_config.py

import os
from dotenv import load_dotenv

# === 설정 ===
load_dotenv()  # 환경 변수 로드
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # 환경 변수에서 Google API 키 가져오기

# 데이터 디렉토리 설정
DATA_DIR = "data"  # 데이터 파일이 저장된 디렉토리

# 모델 설정
EMBEDDING_MODEL = "models/embedding-001"  # 임베딩 모델
CHAT_MODEL = "gemini-1.5-flash"  # 채팅 모델

# 청크 설정
CHUNK_SIZE = 800  # 청크 크기
CHUNK_OVERLAP = 300  # 청크 겹침 크기
TOP_K = 4  # 검색할 문서의 수
