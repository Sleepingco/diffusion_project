# image_processing.py
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import streamlit as st

def segment_anime_image(model, pil_img, device, threshold=0.67):
    """
    Anime-seg를 사용하여 캐릭터가 분리된 RGBA 이미지를 반환합니다.
    (비율 유지, 패딩, 정확한 알파 채널 적용)
    """
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    orig_w, orig_h = pil_img.size
    model_input_size = 1024
    ratio = model_input_size / max(orig_w, orig_h)
    resized_w, resized_h = int(orig_w * ratio), int(orig_h * ratio)
    resized_image = pil_img.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    padded_image = Image.new(
        "RGB", (model_input_size, model_input_size), (255, 255, 255)
    )
    paste_x = (model_input_size - resized_w) // 2
    paste_y = (model_input_size - resized_h) // 2
    padded_image.paste(resized_image, (paste_x, paste_y))
    input_tensor = to_tensor(padded_image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)
        mask = torch.sigmoid(pred[0][0]).cpu().numpy()

    mask_cropped = mask[paste_y : paste_y + resized_h, paste_x : paste_x + resized_w]
    mask_resized = cv2.resize(
        mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )
    alpha = np.where(mask_resized > threshold, 255, 0).astype(np.uint8)
    rgba_img = pil_img.convert("RGBA")
    rgba_array = np.array(rgba_img)
    rgba_array[:, :, 3] = alpha
    result_img = Image.fromarray(rgba_array, "RGBA")
    return result_img


def composite_with_background(character_rgba, background_path, character_scale=0.8):
    """
    RGBA 캐릭터 이미지를 배경 전체에 중앙 하단(바닥) 배치로 합성합니다.
    """
    try:
        # 배경 이미지 로드 및 RGB 변환
        bg_img = Image.open(background_path).convert("RGB")
        bg_width, bg_height = bg_img.size
        print(f"배경 이미지 크기: {bg_width}x{bg_height}")

        # 캐릭터 이미지 RGBA 모드 확인 및 변환
        if character_rgba.mode != "RGBA":
            print(f"캐릭터 이미지 모드를 {character_rgba.mode}에서 RGBA로 변환")
            character_rgba = character_rgba.convert("RGBA")

        char_width, char_height = character_rgba.size
        print(f"캐릭터 이미지 크기: {char_width}x{char_height}")

        # 캐릭터를 배경 크기에 맞춰 스케일링 (비율 유지)
        target_char_height = int(bg_height * character_scale)
        scale_ratio = target_char_height / char_height
        target_char_width = int(char_width * scale_ratio)

        print(
            f"캐릭터 리사이즈: {target_char_width}x{target_char_height} (비율: {scale_ratio:.3f})"
        )

        # 캐릭터 리사이즈
        character_resized = character_rgba.resize(
            (target_char_width, target_char_height), Image.Resampling.LANCZOS
        )

        # 캐릭터 배치 위치 계산 (가로: 중앙, 세로: 바닥에 붙음)
        paste_x = (bg_width - target_char_width) // 2
        paste_y = bg_height - target_char_height  # 바닥에 완전히 붙임

        print(f"캐릭터 배치 위치: x={paste_x}, y={paste_y} (바닥 붙임)")

        # 배경 이미지 복사본 생성
        result_img = bg_img.copy()

        # 캐릭터 합성 (알파 채널 사용)
        if (
            paste_x >= 0
            and paste_y >= 0
            and paste_x + target_char_width <= bg_width
            and paste_y + target_char_height <= bg_height
        ):
            result_img.paste(character_resized, (paste_x, paste_y), character_resized)
            print("캐릭터 합성 완료")
        else:
            print(f"경고: 캐릭터가 배경 범위를 벗어남. 중앙에 배치합니다.")
            paste_x = (bg_width - target_char_width) // 2
            paste_y = (bg_height - target_char_height) // 2
            result_img.paste(character_resized, (paste_x, paste_y), character_resized)

        return result_img

    except Exception as e:
        print(f"배경 합성 중 오류: {e}")
        print(f"배경 경로: {background_path}")
        print(
            f"캐릭터 이미지 모드: {character_rgba.mode if character_rgba else 'None'}"
        )
        print(
            f"캐릭터 이미지 크기: {character_rgba.size if character_rgba else 'None'}"
        )
        raise e


def segment_person_with_mediapipe(segmenter, pil_img, threshold=0.5):
    """
    MediaPipe를 사용하여 원본 이미지에서 인물을 분리한 RGBA 이미지를 반환합니다.
    """
    # PIL 이미지를 NumPy 배열로 변환 (RGB)
    img_np = np.array(pil_img.convert("RGB"))
    
    # MediaPipe로 처리
    results = segmenter.process(img_np)
    
    # 마스크 생성 (결과 > threshold 이면 True)
    mask_condition = results.segmentation_mask > threshold
    
    # 3채널 RGBA 이미지 생성 준비
    # 원본 이미지와 동일한 크기의 투명 배경 (0, 0, 0, 0) 생성
    rgba_img = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
    
    # 원본 이미지의 RGB 값을 복사
    rgba_img[:, :, :3] = img_np
    
    # 마스크를 기반으로 알파 채널 설정 (인물 부분: 255, 배경: 0)
    rgba_img[:, :, 3] = np.where(mask_condition, 255, 0)
    
    # NumPy 배열을 다시 PIL 이미지로 변환
    return Image.fromarray(rgba_img, "RGBA")


def calculate_composition_geometry(char_size, bg_size, character_scale=0.8):
    """
    캐릭터와 배경 크기를 기반으로 합성 위치(x, y)와 리사이즈될 크기(w, h)를 계산합니다.
    composite_with_background 함수의 계산 로직과 동일합니다.
    """
    char_width, char_height = char_size
    bg_width, bg_height = bg_size

    # 목표 캐릭터 높이 계산
    target_char_height = int(bg_height * character_scale)
    scale_ratio = target_char_height / char_height
    target_char_width = int(char_width * scale_ratio)
    
    # 배치 위치 계산 (중앙 하단)
    paste_x = (bg_width - target_char_width) // 2
    paste_y = bg_height - target_char_height
    
    return {
        "pos": (paste_x, paste_y),
        "size": (target_char_width, target_char_height)
    }

def preprocess_control_images(annotators, input_image):
    """ControlNet용 Pose, Canny 맵을 생성합니다."""
    with st.spinner("ControlNet 맵 생성 중..."):
        pose_map = annotators["pose"](input_image)
        canny_map = annotators["canny"](input_image)
        torch.cuda.empty_cache()
    return {"pose": pose_map, "canny": canny_map}