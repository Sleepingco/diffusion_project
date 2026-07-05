# app.py
import streamlit as st
import torch, random, time, traceback
from PIL import Image, ImageFilter
from io import BytesIO

# --- 모듈화된 파일에서 함수 가져오기 ---
from config import CONFIG
from ui import setup_sidebar, display_results, clear_results
from model_loader import load_pipeline, load_anime_segmenter, load_mediapipe_segmenter
from image_processing import (
    preprocess_control_images,
    segment_anime_image,
    composite_with_background,
    segment_person_with_mediapipe,
    calculate_composition_geometry,
)
from utils import (
    get_stage_weights,
    update_progress_bar,
    complete_progress_bar,
    get_new_dimensions,
)

try:
    from lottie_utils import show_lottie_animation
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

# ---------- 1. 페이지 및 세션 상태 설정 ----------
st.set_page_config(layout="wide", page_title="AI Image Generator")
st.title("🎨 AI 필터 스튜디오")
with st.expander("🤔 사용 가이드 (클릭하여 펼쳐보세요!)", expanded=True):
    st.markdown(
        """
        **AI 필터 스튜디오에 오신 것을 환영합니다! 간단한 단계로 여러분의 사진을 다른 스타일로 바꿔보세요.**

        1.  **🎨 필터 선택:** 왼쪽 사이드바에서 마음에 드는 AI 필터 (예: 지브리, 픽사 스타일)를 선택하세요.
        2.  **🖼️ 이미지 업로드:** 변환하고 싶은 **인물 사진**을 업로드합니다. 얼굴과 전신이 잘 보일수록 결과가 좋습니다.
        3.  **✨ 합성 모드 (선택 사항):**
            * **`사용 안 함`**: 업로드한 사진 구도 그대로 이미지를 AI 필터로 변환합니다.
            * **`배경으로 합성`**: AI가 생성한 캐릭터만 분리하여, 미리 준비된 멋진 배경과 합쳐줍니다.
        4.  **🚀 생성 시작:** 설정이 완료되었다면 **'이미지 생성 시작!'** 버튼을 클릭하세요! 잠시 기다리면 새로운 이미지가 생성됩니다.
        5.  **💾 저장 및 공유:** 완성된 이미지를 확인하고 다운로드하여 친구들과 공유해보세요!
        """
    )
st.markdown("---")

if "generation_in_progress" not in st.session_state:
    st.session_state.generation_in_progress = False
if "current_filter" not in st.session_state:
    st.session_state.current_filter = list(CONFIG["filters"].keys())[0]
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "사용 안 함"

# ---------- 2. UI 및 모델 로딩 ----------
ui = setup_sidebar()
lottie_placeholder = st.empty()

# 1. 필터가 변경되었을 때: 모델 전체를 새로 로드해야 하므로 캐시를 비웁니다.
if st.session_state.current_filter != ui["filter_name"]:
    st.toast(f"필터를 '{ui['filter_name']}'(으)로 변경합니다. 모델을 새로 로드합니다...")
    st.session_state.current_filter = ui["filter_name"]
    st.cache_resource.clear()  # 전체 캐시 삭제
    clear_results()

# 2. 합성 모드가 변경되었을 때: 이전 결과만 지우고, 모델은 그대로 둡니다.
if st.session_state.current_mode != ui["mode"]:
    st.toast(f"합성 모드를 '{ui['mode']}'(으)로 변경합니다.")
    st.session_state.current_mode = ui["mode"]
    clear_results()
    st.rerun()

# 모델 로딩
try:
    if LOTTIE_AVAILABLE:
        with lottie_placeholder:
            show_lottie_animation(filepath="Man and robot.json", height=300, key="rocket")
            
    t0 = time.perf_counter()
    pipe, annotators, lora_status, device = load_pipeline(ui["filter_config"])
    anime_segmenter = load_anime_segmenter(device) if ui["mode"] == "배경으로 합성" else None
    mediapipe_segmenter = load_mediapipe_segmenter() if ui["mode"] == "배경으로 합성" else None
    t1 = time.perf_counter()

    lottie_placeholder.empty()
    st.sidebar.success(f"**모델 로딩 완료!** ({t1 - t0:.2f}초)")

except Exception as e:
    lottie_placeholder.empty()
    st.error(f"치명적 오류: 모델 로딩 실패\n\n{e}")
    st.stop()


# ---------- 3. 메인 로직 ----------

# 3-A. 시작 전 화면
if not st.session_state.get("generation_in_progress") and "result_image" not in st.session_state:
    if ui["uploaded_file"]:
        input_original = Image.open(ui["uploaded_file"]).convert("RGB")
        st.subheader("🖼️ 업로드된 원본 이미지")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(input_original, use_container_width=True)
        st.info("👈 사이드바에서 설정을 완료한 후 '이미지 생성 시작!' 버튼을 눌러주세요.")
    else:
        st.info("👈 사이드바에서 인물 이미지를 업로드해주세요.")

# 3-B. 생성 프로세스
elif st.session_state.get("generation_in_progress"):
    try:
        # --- 초기화 ---
        st.subheader("✨ 생성 중...")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            placeholder = st.empty()
        progress_bar = st.progress(0, text="대기 중...")
        stage_weights = get_stage_weights(ui["mode"])
        completed_stages = []
        
        # --- 미리보기 이미지 준비 ---
        input_original = Image.open(st.session_state.uploaded_file).convert("RGB")
        nw, nh = get_new_dimensions(input_original.size)
        input_for_generation = input_original.resize((nw, nh), Image.Resampling.LANCZOS)
        
        if ui["mode"] == "배경으로 합성" and ui["bg_path"] and mediapipe_segmenter:
            bg_img = Image.open(ui["bg_path"]).convert("RGB")
            segmented_person = segment_person_with_mediapipe(mediapipe_segmenter, input_original)
            geom = calculate_composition_geometry(segmented_person.size, bg_img.size)
            person_resized = segmented_person.resize(geom["size"], Image.Resampling.LANCZOS)
            composite_preview = bg_img.copy()
            composite_preview.paste(person_resized, geom["pos"], person_resized)
            blurred_preview = composite_preview.filter(ImageFilter.GaussianBlur(radius=20))
        else:
            blurred_preview = input_for_generation.filter(ImageFilter.GaussianBlur(radius=20))
        placeholder.image(blurred_preview, caption="생성을 시작합니다...", use_container_width=True)

        t_start = time.perf_counter()

        # --- 1. ControlNet ---
        update_progress_bar(progress_bar, "controlnet", 0.0, stage_weights, completed_stages)
        control_imgs = preprocess_control_images(annotators, input_for_generation)
        update_progress_bar(progress_bar, "controlnet", 1.0, stage_weights, completed_stages)
        completed_stages.append("controlnet")

        # --- 2. AI 이미지 생성 ---
        adv = ui["filter_config"]["advanced_settings"]
        seed = random.randint(0, 2**32 - 1)
        actual_steps = int(adv["steps"] * adv["denoising"])

        def progress_bar_callback(pipe, step_index, timestep, callback_kwargs):
            frac = (step_index + 1) / actual_steps
            current_progress = update_progress_bar(
                progress_bar, "generation", frac, stage_weights, completed_stages
            )
            progress_bar.progress(
                int(current_progress * 100),
                text=f"AI 생성 중... (스텝 {step_index + 1}/{actual_steps})",
            )
            return callback_kwargs

        generated_image = pipe(
            prompt=ui["filter_config"]["trigger"],
            negative_prompt=ui["filter_config"]["negative"],
            image=input_for_generation, height=nh, width=nw,
            control_image=list(control_imgs.values()),
            num_inference_steps=adv["steps"], guidance_scale=adv["cfg"],
            strength=adv["denoising"], cross_attention_kwargs={"scale": adv["lora_weight"]},
            controlnet_conditioning_scale=[adv["pose_scale"], adv["canny_scale"]],
            generator=torch.Generator(device=device).manual_seed(seed),
            callback_on_step_end=progress_bar_callback,
        ).images[0]
        completed_stages.append("generation")

        # --- 3. 배경 합성 ---
        final_result = generated_image
        character_rgba = None
        if ui["mode"] == "배경으로 합성":
            update_progress_bar(progress_bar, "background_process", 0.0, stage_weights, completed_stages)
            character_rgba = segment_anime_image(anime_segmenter, generated_image, device)
            if ui["bg_path"]:
                final_result = composite_with_background(character_rgba, ui["bg_path"])
            else:
                st.warning("배경이 선택되지 않아 투명 배경의 캐릭터 이미지만 반환합니다.")
                final_result = character_rgba
            completed_stages.append("background_process")

        # --- 4. 페이드인 ---
        if final_result.size != blurred_preview.size:
             blurred_preview = blurred_preview.resize(final_result.size, Image.Resampling.LANCZOS)
        
        final_rgb = final_result.convert("RGB") # 블렌딩을 위해 RGB로 변환
        for i in range(16):
            alpha = i / 15
            frame = Image.blend(blurred_preview, final_rgb, alpha)
            placeholder.image(frame, use_container_width=True)
            update_progress_bar(progress_bar, "fade", alpha, stage_weights, completed_stages)
            time.sleep(0.03)

        # --- 5. 최종 정리 ---
        placeholder.image(final_result, use_container_width=True)
        complete_progress_bar(progress_bar)
        time.sleep(0.2)
        progress_bar.empty()
        
        st.session_state.result_image = final_result
        st.session_state.character_rgba = character_rgba
        st.session_state.last_generation_time = time.perf_counter() - t_start
        st.session_state.used_seed = seed

    except Exception as e:
        st.error(f"이미지 생성 중 오류 발생: {e}")
        traceback.print_exc()
    finally:
        st.session_state.generation_in_progress = False
        st.rerun()

# 3-C. 결과 표시
elif "result_image" in st.session_state:
    display_results(ui)