#응답 형식화
# mmu_respones_formatter.py

class ResponseFormatter:
    @staticmethod
    def format_response(response: str) -> str:
        """응답 텍스트를 보기 좋게 포매팅"""
        formatted_sections = []  # 포매팅된 섹션을 저장할 리스트
        
        # 응답을 주요 섹션으로 분리
        sections = response.split('\n')
        current_section = []  # 현재 섹션을 저장할 리스트
        
        for line in sections:
            line = line.strip()  # 줄 앞뒤 공백 제거
            if not line:
               continue  # 빈 줄은 건너뛰기
            
           # 새로운 섹션 시작
            if any(line.startswith(marker) for marker in ['📌', '📋', '📚', '💡']):
                if current_section:
                    formatted_sections.append('\n'.join(current_section))   # 현재 섹션을 포맷된 섹션에 추가
                    current_section = [] # 현재 섹션 초기화
                current_section.append(line) 
                if not line.endswith(':'):  # 섹션 제목이 아닌 경우에만 빈 줄 추가
                    current_section.append('')  # 섹션 내용과 제목 사이 빈 줄 추가
                continue
        
            # 불릿 포인트 처리
            if line.startswith('•'):
                # 여러 불릿 포인트가 한 줄에 있는 경우 분리
                bullet_points = line.split('•')[1:]  # 첫 번째 빈 문자열 제거
                for point in bullet_points:
                    if point.strip():  # 불릿 포인트가 비어있지 않은 경우
                        current_section.append(f"  • {point.strip()}")  # 불릿 포인트 포맷팅
                        current_section.append('')  # 각 불릿 포인트 뒤에 빈 줄 추가
                continue
            
            current_section.append(line)  # 일반적인 줄 추가
    
        # 마지막 섹션 추가
        if current_section:
            formatted_sections.append('\n'.join(current_section))  # 마지막 섹션을 포맷된 섹션에 추가
    
        # 최종 텍스트 조합
        formatted_text = '\n\n'.join(formatted_sections)  # 모든 섹션을 줄바꿈으로 조합
    
        # 중복되는 빈 줄 제거 및 정리
        lines = formatted_text.split('\n')
        cleaned_lines = []  # 정리된 줄을 저장할 리스트
        prev_empty = False  # 이전 줄이 빈 줄인지 여부
    
        for line in lines:
            if line.strip():  # 줄이 비어있지 않은 경우
                cleaned_lines.append(line)  # 정리된 줄에 추가
                prev_empty = False  # 이전 줄은 비어있지 않음
            elif not prev_empty:  # 이전 줄이 비어있지 않았을 경우
                cleaned_lines.append(line)  # 빈 줄 추가
                prev_empty = True  # 이전 줄은 비어있음
    
        return '\n'.join(cleaned_lines)  # 정리된 줄을 문자열로 반환