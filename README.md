# AI 필터 스튜디오

Streamlit 기반의 이미지 변환 웹 애플리케이션입니다. 업로드한 인물 사진을 Stable Diffusion XL, ControlNet, LoRA 필터로 애니메이션·회화 스타일 이미지로 변환하고, 선택적으로 준비된 배경 이미지와 합성할 수 있습니다.

## 주요 기능

- **AI 스타일 필터**: 지브리 스타일, 케이팝 데몬 헌터스, 픽사 애니메이션, 유화 필터 제공
- **ControlNet 기반 구도 유지**: OpenPose와 Canny 전처리를 사용해 입력 이미지의 자세와 윤곽을 반영
- **배경 합성 모드**: 생성된 캐릭터를 분리한 뒤 `backgrounds/` 디렉터리의 배경 이미지와 합성
- **결과 다운로드**: 최종 이미지와 분리된 캐릭터 PNG를 Streamlit UI에서 다운로드
- **모델 캐싱**: Streamlit 캐시를 활용해 모델 로딩 시간을 줄임

## 프로젝트 구조

```text
.
├── app.py                    # Streamlit 앱 진입점 및 생성 플로우
├── ui.py                     # 사이드바, 결과 표시, 다운로드 UI
├── config.py                 # 모델 경로, 필터, LoRA, 프롬프트 설정
├── model_loader.py           # SDXL/ControlNet/세그멘테이션 모델 로더
├── image_processing.py       # 전처리, 세그멘테이션, 배경 합성 로직
├── utils.py                  # 진행률, 이미지 크기, 배경 목록 유틸리티
├── backgrounds/              # 합성용 배경 이미지
├── anime_segmentation_main/  # 캐릭터 분리용 anime-seg 모듈
├── ModelList.txt             # 사용 모델 목록 메모
└── requirements.txt          # Python 의존성
```

## 요구 사항

- Python 3.10 이상 권장
- CUDA 지원 GPU 권장
- PyTorch CUDA 12.8 환경 기준 의존성
- Stable Diffusion XL 체크포인트, VAE, ControlNet, LoRA 파일

> 모델 파일은 용량이 커서 저장소에 포함되어 있지 않을 수 있습니다. `config.py`의 경로와 동일하게 `models/` 디렉터리 아래에 준비해야 합니다.

## 설치 방법

1. 저장소를 클론하고 프로젝트 디렉터리로 이동합니다.

   ```bash
   git clone <repository-url>
   cd diffusion_project
   ```

2. 가상 환경을 생성하고 활성화합니다.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. 의존성을 설치합니다.

   ```bash
   pip install -r requirements.txt
   ```

4. 모델 파일을 준비합니다.

   기본 설정은 다음 구조를 기대합니다.

   ```text
   models/
   ├── annotators/
   ├── checkpoints/
   ├── controlnet/
   ├── loras/
   └── vae/
   ```

   필수 또는 권장 파일명은 `config.py`의 `CONFIG` 값을 확인해 맞춰 주세요.

## 실행 방법

```bash
streamlit run app.py
```

브라우저에서 Streamlit 앱이 열리면 다음 순서로 사용합니다.

1. 사이드바에서 AI 필터를 선택합니다.
2. 변환할 인물 이미지를 업로드합니다.
3. 필요하면 `배경으로 합성` 모드를 선택하고 배경을 고릅니다.
4. `이미지 생성 시작!` 버튼을 클릭합니다.
5. 생성된 결과를 확인하고 다운로드합니다.

## 설정 변경

필터, 프롬프트, LoRA 가중치, ControlNet 스케일, 추론 스텝 등은 `config.py`에서 수정할 수 있습니다.

새 배경 이미지를 추가하려면 `backgrounds/` 디렉터리에 `.png`, `.jpg`, `.jpeg` 파일을 넣으면 앱 사이드바의 배경 선택 목록에 표시됩니다.

## 문제 해결

- **모델 파일을 찾을 수 없음**: `models/` 하위 경로와 파일명이 `config.py` 설정과 일치하는지 확인합니다.
- **CUDA 메모리 부족**: 입력 이미지 크기를 줄이거나 다른 프로세스의 GPU 사용량을 줄입니다.
- **anime-seg 로딩 실패**: `anime_segmentation_main/` 모듈과 관련 의존성이 설치되어 있는지 확인합니다.
- **MediaPipe 오류**: `mediapipe`가 현재 Python 버전과 호환되는지 확인합니다.

## 라이선스

이 저장소의 라이선스가 별도로 명시되어 있지 않다면, 사용 전 소유자에게 이용 조건을 확인하세요. 포함된 서브모듈 또는 외부 모델은 각각의 라이선스를 따릅니다.
