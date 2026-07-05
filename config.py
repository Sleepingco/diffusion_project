from pathlib import Path
import logging

# ---------- 환경설정(CONFIG) ----------
CONFIG = {
    "models_dir": Path("./models"),
    "annotators_dir": Path("./models/annotators"),
    "backgrounds_dir": Path("./backgrounds"),
    "vae_path": "vae/sdxl.vae.safetensors",
    "controlnets": {
        "pose": "controlnet/OpenPoseXL2.safetensors",
        "canny": "controlnet/diffusion_pytorch_model_canny.safetensors",
    },
    "filters": {
        "지브리 스타일": {
            "checkpoint": "checkpoints/animagineXL40_v4Opt.safetensors",
            "lora": "loras/Ghibli_xl_v2.safetensors",
            "trigger": "Studio Ghibli, hand-drawn animation style, masterpiece",
            "negative": "lowres, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, signature, watermark, username, blurry, realistic, photorealistic, 3d, painterly, oil painting, aged, mature face, wrinkles, dark, gritty",
            "advanced_settings": {
                "denoising": 0.8,
                "steps": 10,
                "lora_weight": 1.0,
                "cfg": 5.0,
                "pose_scale": 0.7,
                "canny_scale": 0.6,
            },
        },
        "케이팝 데몬 헌터스": {
            "checkpoint": "checkpoints/sdXL_v10VAEFix.safetensors",
            "lora": "loras/K-pop_Demon_Hunters_Style.safetensors",
            "trigger": "KPDH, CGI, animated, movie, Netflix, K-pop, movie scene",
            "negative": "lowres, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, signature, watermark, username, blurry, realistic, photorealistic, painterly, oil painting, aged, mature face, wrinkles, dark, gritty",
            "advanced_settings": {
                "denoising": 0.6,
                "steps": 10,
                "lora_weight": 1.20,
                "cfg": 7.0,
                "pose_scale": 0.7,
                "canny_scale": 1.0,
            },
        },
        "픽사 애니메이션": {
            "checkpoint": "checkpoints/sdXL_v10VAEFix.safetensors",
            "lora": "loras/PixarXL.safetensors",
            "trigger": "pixar style, 3d animation, masterpiece, ultra-detailed, sharp focus,8k resolution, cinematic lighting, soft rim light, subsurface scattering,vibrant pastel palette, big sparkling eyes, joyful smile, 35mm lens, f/1.4, shallow depth of field",
            "negative": "lowres, bad anatomy, bad hands, missing fingers, extra limbs, poorly drawn face,jpeg artifacts, noisy, blurry, text, signature, watermark, logo, photorealistic,realistic, painterly, oil painting, aged skin, wrinkles, dark, gritty,monster",
            "advanced_settings": {
                "denoising": 0.85,
                "steps": 10,
                "lora_weight": 1.0,
                "cfg": 7.0,
                "pose_scale": 0.7,
                "canny_scale": 0.9,
            },
        },
        "유화": {
            "checkpoint": "checkpoints/envyStarlightXL01Lightning_galaxyV20.safetensors",
            "lora": "loras/envyStarlightThickPalette_01v10.safetensors",
            "trigger": "oil painting, impasto oil painting, thick brushstrokes, rich pigments, palette knife,visible palette knife marks, vibrant yet earthy palette, dramatic chiaroscuro,ultra-high-resolution, masterpiece,highly detailed brush texture",
            "negative": "malformed, bad anatomy, distorted face, extra limbs, missing fingers,flat colors, digital smooth, cg render, vector art, cartoon,lowres, low detail, blurry, grainy, jpeg artifacts, muted colors",
            "advanced_settings": {
                "denoising": 0.6,
                "steps": 10,
                "lora_weight": 1.0,
                "cfg": 6.0,
                "pose_scale": 0.7,
                "canny_scale": 1.0,
            },
        },
    },
}

def validate_config():
    """설정값의 유효성을 검증합니다."""
    issues = []
    
    # 디렉토리 존재 확인
    if not CONFIG["models_dir"].exists():
        issues.append(f"모델 디렉토리가 존재하지 않습니다: {CONFIG['models_dir']}")
    
    # 필수 파일 확인
    essential_files = [CONFIG["vae_path"]] + list(CONFIG["controlnets"].values())
    for file_path in essential_files:
        full_path = CONFIG["models_dir"] / file_path
        if not full_path.exists():
            issues.append(f"필수 파일이 없습니다: {full_path}")
    
    # 필터 설정 검증
    for filter_name, filter_config in CONFIG["filters"].items():
        required_keys = ["checkpoint", "trigger", "negative", "advanced_settings"]
        for key in required_keys:
            if key not in filter_config:
                issues.append(f"필터 '{filter_name}'에 '{key}' 설정이 없습니다.")
        
        # 체크포인트 파일 존재 확인
        if "checkpoint" in filter_config:
            checkpoint_path = CONFIG["models_dir"] / filter_config["checkpoint"]
            if not checkpoint_path.exists():
                issues.append(f"필터 '{filter_name}'의 체크포인트가 없습니다: {checkpoint_path}")
    
    if issues:
        logging.warning(f"설정 검증에서 {len(issues)}개의 문제를 발견했습니다:")
        for issue in issues:
            logging.warning(f"  - {issue}")
    
    return len(issues) == 0

# 설정 검증 실행 (import 시점에)
if __name__ != "__main__":
    validate_config()