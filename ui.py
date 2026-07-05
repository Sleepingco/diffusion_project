# ui.py
import streamlit as st
from io import BytesIO
from PIL import Image

from config import CONFIG
from utils import get_background_options

def clear_results():
    """세션 상태에서 이전 생성 결과를 모두 삭제합니다."""
    keys_to_delete = ["result_image", "character_rgba", "last_generation_time", "used_seed"]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.generation_in_progress = False

def start_generation():
    """생성 시작 플래그를 설정합니다."""
    if st.session_state.get("uploaded_file") is not None:
        st.session_state.generation_in_progress = True

def setup_sidebar():
    """사이드바 UI를 구성하고 사용자 입력을 반환합니다."""
    with st.sidebar:
        st.header("⚙️ 생성 설정")
        filter_name = st.selectbox(
            "1. 필터 선택",
            options=list(CONFIG["filters"].keys()),
            key="selected_filter",
        )
        uploaded_file = st.file_uploader(
            "2. 인물 이미지 업로드", type=["png", "jpg", "jpeg"], key="uploaded_file"
        )

        st.markdown("---")
        st.header("✨ 합성 모드 선택")
        mode = st.radio(
            "어떤 방식으로 합성할까요?",
            ["사용 안 함", "배경으로 합성"],
            key="composition_mode",
            help="배경으로 합성: 생성된 이미지에서 캐릭터만 분리하여 선택한 배경과 합성",
        )

        bg_path = None
        if mode == "배경으로 합성":
            st.subheader("🌄 배경 선택")
            st.info("AI 생성된 이미지에서 캐릭터만 분리하여 이 배경과 합성됩니다.")
            background_options = get_background_options(CONFIG["backgrounds_dir"])
            bg_name = st.selectbox(
                "배경 선택", options=list(background_options.keys()), key="background"
            )
            bg_path = background_options.get(bg_name)
            if bg_path:
                st.image(str(bg_path), caption="선택된 배경", use_container_width=True)

        st.markdown("---")
        st.button(
            "🚀 이미지 생성 시작!",
            type="primary",
            use_container_width=True,
            on_click=start_generation,
            disabled=st.session_state.get("generation_in_progress", False),
        )

        return {
            "filter_name": filter_name,
            "filter_config": CONFIG["filters"][filter_name],
            "uploaded_file": uploaded_file,
            "mode": mode,
            "bg_path": bg_path,
        }

def display_results(ui_params):
    """생성된 최종 결과를 화면에 표시하고 다운로드 버튼을 제공합니다."""
    st.subheader("✨ 최종 결과물")
    caption = f"필터: {ui_params['filter_name']} | 모드: {ui_params['mode']} | 생성 시간: {st.session_state.last_generation_time:.2f}초 | Seed: {st.session_state.used_seed}"
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(st.session_state.result_image, caption=caption, use_container_width=True)
        
        # 분리된 캐릭터 보기/다운로드
        if st.session_state.get("character_rgba") and ui_params["mode"] == "배경으로 합성":
            with st.expander("🎭 분리된 캐릭터 보기/다운로드"):
                st.image(st.session_state.character_rgba, caption="분리된 캐릭터", use_container_width=True)
                char_buf = BytesIO()
                st.session_state.character_rgba.save(char_buf, format="PNG")
                st.download_button(
                    "🎭 분리된 캐릭터 다운로드",
                    char_buf.getvalue(),
                    f"character_{ui_params['filter_name']}_{st.session_state.used_seed}.png",
                    "image/png",
                    use_container_width=True,
                )

        # 최종 결과 다운로드
        buf = BytesIO()
        save_format = "PNG" if st.session_state.result_image.mode == "RGBA" else "JPEG"
        st.session_state.result_image.save(buf, format=save_format)
        file_extension = "png" if save_format == "PNG" else "jpg"

        st.download_button(
            "💾 최종 결과 다운로드",
            buf.getvalue(),
            f"result_{ui_params['filter_name']}_{st.session_state.used_seed}.{file_extension}",
            f"image/{file_extension}",
            use_container_width=True,
        )

        if st.button("🔄 새로운 이미지 생성", type="secondary", use_container_width=True):
            clear_results()
            st.rerun()