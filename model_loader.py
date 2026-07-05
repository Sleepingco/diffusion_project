import streamlit as st
import torch
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from controlnet_aux import OpenposeDetector, CannyDetector
import logging

try:
    from anime_segmentation_main.train import AnimeSegmentation
    ANIME_SEG_AVAILABLE = True
except ImportError:
    ANIME_SEG_AVAILABLE = False
    logging.warning("anime_segmentation_main을 찾을 수 없습니다.")

import mediapipe as mp
from config import CONFIG

class ModelLoadError(Exception):
    """모델 로딩 관련 예외"""
    pass

def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

@st.cache_resource(show_spinner=False)
def load_annotators(annotator_path, device):
    """어노테이터를 로드합니다."""
    try:
        return {
            "pose": OpenposeDetector.from_pretrained(annotator_path).to(device),
            "canny": CannyDetector(),
        }
    except Exception as e:
        raise ModelLoadError(f"어노테이터 로딩 실패: {e}")

@st.cache_resource(show_spinner=False)
def load_pipeline(filter_config):
    """AI 파이프라인을 로드합니다."""
    try:
        md = CONFIG["models_dir"]
        ts = torch.float16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 필수 파일 존재 확인
        checkpoint_path = md / filter_config["checkpoint"]
        if not checkpoint_path.exists():
            raise ModelLoadError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
            checkpoint_path,
            controlnet=[
                ControlNetModel.from_single_file(md / p, torch_dtype=ts)
                for p in CONFIG["controlnets"].values()
            ],
            vae=AutoencoderKL.from_single_file(md / CONFIG["vae_path"], torch_dtype=ts),
            use_safetensors=True,
            torch_dtype=ts,
        )
        
        # LoRA 로딩 (선택적)
        lora_status = "LoRA 없음"
        if "lora" in filter_config:
            try:
                lora_path = md / filter_config["lora"]
                if lora_path.exists():
                    pipe.load_lora_weights(lora_path.parent, weight_name=lora_path.name)
                    lora_status = f"✅ LoRA 로드 성공: {lora_path.name}"
                else:
                    lora_status = f"⚠️ LoRA 파일을 찾을 수 없음: {lora_path.name}"
            except Exception as e:
                lora_status = f"⚠️ LoRA 로드 실패: {e}"
        
        pipe.enable_model_cpu_offload(gpu_id=0)
        pipe.safety_checker = dummy_checker
        pipe.enable_vae_tiling()
        
        annotators = load_annotators(CONFIG["annotators_dir"], device)
        
        return pipe, annotators, lora_status, device
        
    except Exception as e:
        raise ModelLoadError(f"파이프라인 로딩 실패: {e}")

@st.cache_resource(show_spinner=False)
def load_anime_segmenter(device):
    """Anime Segmentation 모델을 로드합니다."""
    if not ANIME_SEG_AVAILABLE:
        raise ModelLoadError("anime_segmentation_main 모듈을 사용할 수 없습니다.")
    
    try:
        model = AnimeSegmentation.from_pretrained("skytnt/anime-seg").to(device)
        model.eval()
        return model
    except Exception as e:
        raise ModelLoadError(f"Anime segmenter 로딩 실패: {e}")

@st.cache_resource(show_spinner=False)
def load_mediapipe_segmenter():
    """MediaPipe Segmentation 모델을 로드합니다."""
    try:
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        return mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
    except Exception as e:
        raise ModelLoadError(f"MediaPipe segmenter 로딩 실패: {e}")