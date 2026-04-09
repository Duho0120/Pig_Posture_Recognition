"""
베이스라인 노트북(pig-posture-recognition-baseline.ipynb)의
Cell 11, 12, 15, 17번을 수정하여 3가지 개선사항을 적용한
Method1_Improved.ipynb를 생성하는 스크립트
"""
import json
import copy

INPUT_PATH  = r"C:\Users\ASUS\Desktop\[ZB]멘토링_프로젝트\Kaggle\Pig_Posture_Recognition\notebookes\pig-posture-recognition-baseline.ipynb"
OUTPUT_PATH = r"C:\Users\ASUS\Desktop\[ZB]멘토링_프로젝트\Kaggle\Pig_Posture_Recognition\notebookes\Method1_Improved.ipynb"

# ============================================================
# 새로 교체할 셀 코드들 (3가지 개선사항 반영)
# ============================================================

# ── 개선 1 + 3: letterbox_resize 함수 (새 셀로 추가) ─────────
CELL_LETTERBOX = """\
# ============================================================
# [개선 1] Letterbox Resize - 바운딩 박스 비율 왜곡 해결
# ============================================================
# 기존: cv2.resize() 로 강제 정사각형 -> 돼지 이미지 찌그러짐
# 개선: 긴 쪽을 기준으로 축소 후 검정 패딩으로 남은 공간 채우기

def letterbox_resize(image: "np.ndarray", target_size: int) -> "np.ndarray":
    \"\"\"
    원본 비율(Aspect Ratio)을 유지하며 target_size x target_size 로 리사이즈.
    빈 공간은 검정(0, 0, 0) 패딩으로 채웁니다.
    \"\"\"
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 검정 캔버스 생성 후 중앙에 배치
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    top  = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas

print("[개선 1] letterbox_resize 함수 정의 완료")

# ---- 비교 시각화 (기존 강제 리사이즈 vs Letterbox) ----
try:
    _sample = train_df.iloc[0]
    _img = cv2.cvtColor(cv2.imread(
        os.path.join(CFG.INPUT_DIR, "train_images", _sample["image_id"])
    ), cv2.COLOR_BGR2RGB)
    _x, _y = max(0, int(_sample["xmin"])), max(0, int(_sample["ymin"]))
    _x2 = min(_img.shape[1], int(_sample["xmin"] + _sample["width_bbox"]))
    _y2 = min(_img.shape[0], int(_sample["ymin"] + _sample["height_bbox"]))
    _crop = _img[_y:_y2, _x:_x2]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cv2.resize(_crop, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)))
    axes[0].set_title(f"[기존] 강제 Resize\\n원본비율:{_crop.shape[1]/max(_crop.shape[0],1):.2f} -> 왜곡")
    axes[0].axis("off")
    axes[1].imshow(letterbox_resize(_crop, CFG.IMAGE_SIZE))
    axes[1].set_title(f"[개선] Letterbox Resize\\n비율 유지: {_crop.shape[1]/max(_crop.shape[0],1):.2f}")
    axes[1].axis("off")
    plt.suptitle("리사이즈 방식 비교", fontweight="bold")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"시각화 건너뜀: {e}")
"""

# ── 개선 1 + 3: Dataset 클래스 (Cell 11 교체) ────────────────
CELL_DATASET = """\
import copy

# ============================================================
# [개선 1] Letterbox Resize + [개선 3] Edge 맵 채널 추가
# ============================================================
# CFG 에 추가 항목
CFG.USE_EDGE_MAP     = True   # True: 4채널(RGBE), False: 기존 3채널(RGB)
CFG.EDGE_THRESHOLD1  = 50     # Canny 하한 임계값
CFG.EDGE_THRESHOLD2  = 150    # Canny 상한 임계값

class PigPostureDataset(Dataset):
    \"\"\"
    [개선 1] letterbox_resize 로 비율 보존
    [개선 3] Canny Edge 채널 추가 -> 4채널(RGBE) 텐서 반환
    \"\"\"
    def __init__(self, dataframe, image_dir, transform=None, is_test=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test   = is_test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_id"])

        # 이미지 로드 (OpenCV: PIL 보다 빠름)
        try:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"이미지 로드 오류 {image_path}: {e}")
            image = np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)

        # 안전한 Bounding Box Crop
        H, W = image.shape[:2]
        xmin = max(0, int(row["xmin"]))
        ymin = max(0, int(row["ymin"]))
        xmax = min(W, int(row["xmin"] + row["width_bbox"]))
        ymax = min(H, int(row["ymin"] + row["height_bbox"]))

        cropped = image[ymin:ymax, xmin:xmax]
        if cropped.size == 0:
            cropped = image

        # [개선 1] Letterbox Resize (비율 왜곡 해결)
        cropped = letterbox_resize(cropped, CFG.IMAGE_SIZE)

        # [개선 3] Edge 채널 생성 (Albumentations 적용 전, 원본 픽셀 기준)
        if CFG.USE_EDGE_MAP:
            gray  = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, CFG.EDGE_THRESHOLD1, CFG.EDGE_THRESHOLD2)

        # 데이터 증강 Transform 적용 (RGB 이미지에만)
        if self.transform:
            rgb_tensor = self.transform(image=cropped)["image"]  # (3, H, W)
        else:
            norm = cropped.astype(np.float32) / 255.0
            rgb_tensor = torch.from_numpy(norm).permute(2, 0, 1)

        # [개선 3] Edge 채널을 4번째 채널로 연결
        if CFG.USE_EDGE_MAP:
            edge_tensor = torch.from_numpy(
                edges.astype(np.float32) / 255.0
            ).unsqueeze(0)                                        # (1, H, W)
            final_tensor = torch.cat([rgb_tensor, edge_tensor], dim=0)  # (4, H, W)
        else:
            final_tensor = rgb_tensor                             # (3, H, W)

        if self.is_test:
            return final_tensor, row["row_id"]
        else:
            return final_tensor, int(row["class_id"])

print("PigPostureDataset (개선 버전) 정의 완료")
print(f"  출력 채널: {4 if CFG.USE_EDGE_MAP else 3}ch  |  USE_EDGE_MAP={CFG.USE_EDGE_MAP}")
"""

# ── 개선 2: 강화된 train_transforms (Cell 12 교체) ───────────
CELL_TRANSFORMS = """\
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================
# [개선 2] CoarseDropout 강화 - Random Erasing 효과
# ============================================================
# 기존: max_holes=8, max_height=32, p=0.3
# 개선: max_holes=16, max_height=48, min 범위 추가, p=0.5
# -> 모델이 이미지 중앙에만 의존하지 않고 전체 영역에서 특징 학습

train_transforms = A.Compose([
    A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),

    # 기하학적 증강
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),

    # 색상 증강
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

    # 노이즈
    A.GaussNoise(p=0.1),
    A.MotionBlur(p=0.1),

    # [개선 2] CoarseDropout 파라미터 강화
    A.CoarseDropout(
        max_holes=16,          # 기존 8 -> 16
        max_height=48,         # 기존 32 -> 48
        max_width=48,          # 기존 32 -> 48
        min_holes=4,           # 최소 구멍 수 추가
        min_height=16,         # 최소 크기 추가
        min_width=16,
        fill_value=0,
        p=0.5                  # 기존 0.3 -> 0.5
    ),

    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

print("[개선 2] Train Transforms 정의 완료 (CoarseDropout 강화)")
print(f"  Train 연산 수: {len(train_transforms)}")
print(f"  Val   연산 수: {len(val_transforms)}")
"""

# ── 개선 3: 4채널 EfficientNetV2Model (Cell 15 교체) ─────────
CELL_MODEL = """\
# ============================================================
# [개선 3] 4채널 입력 EfficientNetV2-S 모델
# ============================================================
# ImageNet 사전학습 가중치(3채널)를 최대한 보존하면서
# 첫 Conv 레이어만 4채널로 확장합니다.
# Edge 채널 가중치는 기존 RGB 3채널 평균값으로 초기화합니다.

class EfficientNetV2Model(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, use_edge_map=True):
        super(EfficientNetV2Model, self).__init__()
        self.use_edge_map = use_edge_map

        # EfficientNetV2-S 사전학습 모델 로드
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_v2_s(weights=weights)

        # [개선 3] 첫 Conv 레이어를 3ch -> 4ch 로 교체
        if use_edge_map:
            old_conv   = self.backbone.features[0][0]
            old_weight = old_conv.weight.data  # (C_out, 3, kH, kW)

            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )

            with torch.no_grad():
                # 기존 RGB 가중치 그대로 복사
                new_conv.weight[:, :3, :, :] = old_weight
                # Edge 채널은 RGB 평균으로 부드럽게 초기화
                new_conv.weight[:, 3:, :, :] = old_weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias

            self.backbone.features[0][0] = new_conv
            print("[개선 3] 첫 Conv: 3ch -> 4ch 확장 완료 (RGB 가중치 보존)")

        # 분류기 헤드 교체
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ---- Shape 검증 (Sanity Check) ----
_in_ch = 4 if CFG.USE_EDGE_MAP else 3
_dummy = torch.randn(2, _in_ch, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)
_m     = EfficientNetV2Model(num_classes=CFG.NUM_CLASSES, pretrained=False,
                              use_edge_map=CFG.USE_EDGE_MAP)
with torch.no_grad():
    _out = _m(_dummy)
print(f"Shape 검증 -> 입력: {_dummy.shape}  /  출력: {_out.shape}")
del _m, _dummy, _out
print("Method 1 (개선 버전): EfficientNetV2-S 정의 완료")
"""

# ── 데이터셋/DataLoader 준비 (Cell 17 교체) ──────────────────
CELL_DATALOADER = """\
print("=== METHOD 1 (개선 버전): EfficientNetV2-S Training ===")

# 전체 Dataset 생성 (transform 은 None 으로 시작)
base_dataset = PigPostureDataset(
    train_df,
    os.path.join(CFG.INPUT_DIR, "train_images"),
    transform=None
)

# 8:2 Train/Val 분리 (클래스 비율 유지)
train_indices, val_indices = train_test_split(
    range(len(train_df)),
    test_size=0.2,
    stratify=train_df["class_id"],
    random_state=CFG.SEED
)

# [버그 수정] deepcopy 로 완전히 분리하여 transform 교차오염 방지
train_ds = copy.deepcopy(base_dataset)
train_ds.transform = train_transforms

val_ds = copy.deepcopy(base_dataset)
val_ds.transform = val_transforms

train_subset = torch.utils.data.Subset(train_ds, train_indices)
val_subset   = torch.utils.data.Subset(val_ds,   val_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_subset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Train 샘플: {len(train_subset)}  |  Val 샘플: {len(val_subset)}")

# DataLoader 배치 Shape 확인
_s_img, _s_lbl = next(iter(train_loader))
print(f"배치 이미지 Shape: {_s_img.shape}   <- (Batch, {4 if CFG.USE_EDGE_MAP else 3}ch, H, W)")
print(f"배치 라벨  Shape: {_s_lbl.shape}")
del _s_img, _s_lbl
print("[OK] DataLoader 검증 완료!")
"""

# ── CFG LEARNING_RATE 패치용 소스 ────────────────────────────
CELL_CFG_PATCH = """\
# ============================================================
# [개선 3 적용을 위한 설정 패치]
# Edge 채널 추가로 초반 학습이 약간 불안정할 수 있으므로
# Learning Rate 를 1e-3 -> 5e-4 로 낮춰 안정성 확보
# ============================================================
CFG.LEARNING_RATE = 5e-4
CFG.BATCH_SIZE    = 64   # A100 OOM 방지
CFG.GRADIENT_ACCUMULATION_STEPS = 2  # 실질 배치: 64 * 2 = 128

print(f"[패치] LR={CFG.LEARNING_RATE}  BATCH={CFG.BATCH_SIZE}  ACCUM={CFG.GRADIENT_ACCUMULATION_STEPS}")
"""

# ── 모델 초기화 (Cell 18 교체) ────────────────────────────────
CELL_INIT = """\
# 모델 생성 (4채널 EfficientNetV2-S)
method1_model = EfficientNetV2Model(
    num_classes=CFG.NUM_CLASSES,
    pretrained=True,
    use_edge_map=CFG.USE_EDGE_MAP
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=CFG.LABEL_SMOOTHING
)

optimizer = optim.AdamW(
    method1_model.parameters(),
    lr=CFG.LEARNING_RATE,
    weight_decay=CFG.WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CFG.NUM_EPOCHS,
    eta_min=1e-6
)

print("Method 1 (개선 버전) 모델 초기화 완료")
print(f"  Mixed Precision     : {CFG.USE_MIXED_PRECISION}")
print(f"  Label Smoothing     : {CFG.LABEL_SMOOTHING}")
print(f"  Learning Rate       : {CFG.LEARNING_RATE}")
print(f"  Batch (실질)        : {CFG.BATCH_SIZE * CFG.GRADIENT_ACCUMULATION_STEPS}")
print(f"  Edge Map 사용       : {CFG.USE_EDGE_MAP}")
"""

# ── 모델 저장 (마지막에 추가할 셀) ───────────────────────────
CELL_SAVE = """\
# ============================================================
# 훈련된 모델 가중치 저장
# ============================================================
import os

# Google Colab 사용 시 드라이브 경로로 변경하세요:
# save_dir = "/content/drive/MyDrive/PigPosture_Models"
save_dir = CFG.OUTPUT_DIR

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "method1_improved_efficientnetv2_best.pth")
torch.save(method1_state, save_path)

print(f"[저장 완료] {save_path}")
print(f"최고 Validation Macro F1: {max(method1_val_f1):.4f}")
print()
print("적용된 개선 사항:")
print("  [개선 1] Letterbox Resize    - 바운딩 박스 비율 유지")
print("  [개선 2] CoarseDropout 강화  - Random Erasing 효과")
print("  [개선 3] Edge 맵 4채널 입력  - 윤곽선 정보 직접 학습")
print("  [버그 수정] deepcopy Train/Val 분리")
"""

# ============================================================
# 노트북 JSON 파싱 및 셀 교체
# ============================================================
def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def make_md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

with open(INPUT_PATH, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── 각 셀의 소스를 하나의 문자열로 합침 ──────────────────────
def src(cell) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s

# ── 특정 키워드로 셀 인덱스 찾기 ─────────────────────────────
def find_cell(keyword: str) -> int:
    for i, c in enumerate(cells):
        if keyword in src(c):
            return i
    return -1

# ──────────────────────────────────────────────────────────────
# 1) Cell 11 (Dataset 클래스) 앞에 letterbox 셀 + CFG 패치 셀 삽입
# ──────────────────────────────────────────────────────────────
idx11 = find_cell("class PigPostureDataset")
assert idx11 != -1, "Dataset 셀을 찾지 못했습니다."

cells[idx11] = make_code_cell(CELL_DATASET)          # Dataset 교체
cells.insert(idx11, make_code_cell(CELL_CFG_PATCH))   # CFG 패치 삽입
cells.insert(idx11, make_code_cell(CELL_LETTERBOX))   # letterbox 삽입

print(f"[OK] Cell {idx11}: letterbox / CFG patch / Dataset 삽입 완료")

# ──────────────────────────────────────────────────────────────
# 2) Cell 12 (train_transforms) 교체
# ──────────────────────────────────────────────────────────────
idx12 = find_cell("train_transforms = A.Compose")
assert idx12 != -1, "train_transforms 셀을 찾지 못했습니다."
cells[idx12] = make_code_cell(CELL_TRANSFORMS)
print(f"[OK] Cell {idx12}: train_transforms 교체 완료")

# ──────────────────────────────────────────────────────────────
# 3) Cell 15 (EfficientNetV2Model) 교체
# ──────────────────────────────────────────────────────────────
idx15 = find_cell("class EfficientNetV2Model")
assert idx15 != -1, "EfficientNetV2Model 셀을 찾지 못했습니다."
cells[idx15] = make_code_cell(CELL_MODEL)
print(f"[OK] Cell {idx15}: EfficientNetV2Model 교체 완료")

# ──────────────────────────────────────────────────────────────
# 4) Cell 17 (DataLoader 준비) 교체
# ──────────────────────────────────────────────────────────────
idx17 = find_cell("METHOD 1: EfficientNetV2-S Training")
assert idx17 != -1, "DataLoader 셀을 찾지 못했습니다."
cells[idx17] = make_code_cell(CELL_DATALOADER)
print(f"[OK] Cell {idx17}: DataLoader 준비 교체 완료")

# ──────────────────────────────────────────────────────────────
# 5) Cell 18 (model init) 교체
# ──────────────────────────────────────────────────────────────
idx18 = find_cell("Initialize Method 1 model")
assert idx18 != -1, "모델 초기화 셀을 찾지 못했습니다."
cells[idx18] = make_code_cell(CELL_INIT)
print(f"[OK] Cell {idx18}: 모델 초기화 교체 완료")

# ──────────────────────────────────────────────────────────────
# 6) 노트북 제목 셀 업데이트
# ──────────────────────────────────────────────────────────────
idx0 = find_cell("Pig Posture Recognition Competition")
if idx0 != -1:
    cells[idx0] = make_md_cell(
        "# Pig Posture Recognition - Method 1 개선 버전\n\n"
        "기존 베이스라인 대비 **3가지 핵심 개선 사항** 적용:\n\n"
        "1. **[개선 1] Letterbox Resize** : 바운딩 박스 리사이즈 시 종횡비 보존 (찌그러짐 방지)\n"
        "2. **[개선 2] Random Erasing 강화** : CoarseDropout 파라미터 강화 (구멍 8→16, 크기 32→48, 확률 0.3→0.5)\n"
        "3. **[개선 3] Edge 맵 채널 추가** : RGB + Canny Edge = 4채널(RGBE) 입력으로 윤곽선 정보 학습\n\n"
        "추가 버그 수정: `deepcopy`로 Train/Val transform 올바르게 분리\n\n"
        "**목표: Macro F1 Score 향상**"
    )
    print(f"[OK] Cell {idx0}: 제목 업데이트 완료")

# ──────────────────────────────────────────────────────────────
# 7) 마지막에 모델 저장 셀 추가
# ──────────────────────────────────────────────────────────────
cells.append(make_md_cell("## 모델 저장 (Google Drive)"))
cells.append(make_code_cell(CELL_SAVE))
print("[OK] 모델 저장 셀 추가 완료")

# ──────────────────────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────────────────────
nb["cells"] = cells
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print()
print("=" * 60)
print(f"[완료] 개선 노트북 생성: {OUTPUT_PATH}")
print("=" * 60)
