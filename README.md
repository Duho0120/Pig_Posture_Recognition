# Pig Posture Recognition 🐖

EfficientNet 모델을 활용한 돼지 자세 인식 프로젝트입니다. 베이스라인 모델과 다양한 개선 기법(Letterbox Resizing, CoarseDropout 강화, RGBE 4채널 구조)이 적용된 모델의 성능을 비교합니다.

## 프로젝트 개요
이 프로젝트의 주요 목표는 가구동물의 행동 및 상태를 파악하기 위해 인공지능 모델을 학습시키고, 다양한 전처리 및 증강 기법을 통해 성능을 극대화하는 것입니다.

## 주요 기능 및 개선사항
- **Letterbox Resizing**: 이미지 왜곡을 최소화하면서 모델 크기에 맞게 리사이징 히는 기법 적용.
- **CoarseDropout 강화**: 모델의 강건성(Robustness)을 높이기 위한 강력한 Random Erasing 적용.
- **RGBE 4채널 구조**: 구조적 특징을 더 잘 포착하기 위한 입력 채널 확장.
- **성능 비교**: `EfficientNet_비교.ipynb`를 통해 베이스라인 대비 향상된 모델의 F1 Score 및 Confusion Matrix 분석.

## 폴더 구조
- `notebookes/`: 데이터 다운로드, 학습 및 성능 평가를 위한 Jupyter Notebook 파일들.
- `output/`: (로컬 전용) 학습된 모델 가중치 파일들이 저장됩니다. (.gitignore에 의해 GitHub 제외)
- `data/`: (로컬 전용) 학습용 데이터셋 폴더입니다. (.gitignore에 의해 GitHub 제외)

## 시작하기
1. 필요한 라이브러리 설치:
   ```bash
   pip install -r requirements.txt
   ```
2. `notebookes/` 폴더 내의 노트북을 실행하여 모델을 학습시키거나 결과를 비교합니다.

## 기술 스택
- Python, PyTorch, EfficientNetV2, Albumentations, OpenCV, Timm
