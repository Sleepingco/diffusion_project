# lottie_utils.py
import streamlit as st
from streamlit_lottie import st_lottie
import json
import streamlit as st
from streamlit_lottie import st_lottie

def load_lottie_file(filepath: str):
    """
    주어진 로컬 Lottie 애니메이션 JSON 파일을 로드합니다.

    Args:
        filepath (str): Lottie JSON 파일의 로컬 경로

    Returns:
        dict | None: JSON 데이터 또는 로드 실패 시 None
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Lottie 애니메이션 로드 실패: {e}")
        return None


def show_lottie_animation(filepath: str, height: int = 200, key: str = None):
    """
    Streamlit에 Lottie 애니메이션을 표시합니다.

    Args:
        filepath (str): Lottie JSON 파일의 로컬 경로
        height (int): 애니메이션 표시 높이 (픽셀)
        key (str | None): Streamlit 위젯 키
    """
    lottie_json = load_lottie_file(filepath)
    if lottie_json:
        st_lottie(
            lottie_json,
            height=height,
            key=key,
        )
