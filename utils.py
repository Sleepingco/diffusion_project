import streamlit as st
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 상수 정의
DEFAULT_TARGET_AREA = 1024 * 1024
DIMENSION_ALIGNMENT = 64
SUPPORTED_IMAGE_FORMATS = ["*.png", "*.jpg", "*.jpeg"]

# 각 모드별 단계 가중치 정의
MODE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "사용 안 함": {
        "controlnet": 0.15,
        "generation": 0.80,
        "fade": 0.05,
    },
    "배경으로 합성": {
        "controlnet": 0.15,
        "generation": 0.65,
        "background_process": 0.15,
        "fade": 0.05,
    },
}

def get_stage_weights(mode: str) -> Dict[str, float]:
    """현재 모드에 맞는 단계별 가중치를 반환합니다."""
    weights = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["사용 안 함"])
    total = sum(weights.values())
    return {stage: weight / total for stage, weight in weights.items()}

def update_progress_bar(
    pb: st.delta_generator.DeltaGenerator,
    current_stage: str,
    stage_progress: float,
    stage_weights: Dict[str, float],
    completed_stages: Optional[List[str]] = None
) -> float:
    """
    진행률 바를 업데이트합니다.
    
    Args:
        pb: Streamlit progress bar 객체
        current_stage: 현재 실행 중인 단계명
        stage_progress: 현재 단계 내부 진행률 (0.0~1.0)
        stage_weights: 각 단계의 가중치 딕셔너리
        completed_stages: 완료된 단계들의 리스트
        
    Returns:
        전체 진행률 (0.0~1.0)
    """
    if completed_stages is None:
        completed_stages = []
    
    completed_weight = sum(stage_weights.get(stage, 0) for stage in completed_stages)
    current_weight = stage_weights.get(current_stage, 0)
    total_progress = min(completed_weight + (stage_progress * current_weight), 1.0)
    
    pb.progress(int(total_progress * 100))
    return total_progress

def complete_progress_bar(pb: st.delta_generator.DeltaGenerator) -> None:
    """진행률 바를 강제로 100%로 설정합니다."""
    pb.progress(100)

def get_new_dimensions(
    original_size: Tuple[int, int], 
    target_area: int = DEFAULT_TARGET_AREA
) -> Tuple[int, int]:
    """
    원본 비율을 유지하면서 목표 픽셀 수에 가깝게 새 차원을 계산합니다.
    
    Args:
        original_size: 원본 이미지 크기 (width, height)
        target_area: 목표 픽셀 수
        
    Returns:
        새로운 차원 (width, height), 64의 배수로 정렬됨
    """
    ow, oh = original_size
    ar = ow / oh
    nw = int(math.sqrt(target_area * ar))
    nh = int(nw / ar)
    return (
        round(nw / DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT,
        round(nh / DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT
    )

@st.cache_data
def get_background_options(directory: Path) -> Dict[str, Optional[Path]]:
    """
    지정된 디렉토리에서 배경 이미지 목록을 가져옵니다.
    
    Args:
        directory: 배경 이미지가 있는 디렉토리
        
    Returns:
        파일명을 키로 하고 경로를 값으로 하는 딕셔너리
    """
    if not directory.is_dir():
        return {"선택 안 함": None}
    
    paths = []
    for fmt in SUPPORTED_IMAGE_FORMATS:
        paths.extend(directory.glob(fmt))
    
    return {"선택 안 함": None, **{p.name: p for p in sorted(paths)}}

def validate_image_file(file_path: Path) -> bool:
    """
    이미지 파일의 유효성을 검증합니다.
    
    Args:
        file_path: 검증할 파일 경로
        
    Returns:
        유효한 이미지 파일인지 여부
    """
    if not file_path.exists():
        return False
    
    # 확장자 확인
    allowed_extensions = ['.png', '.jpg', '.jpeg']
    return file_path.suffix.lower() in allowed_extensions