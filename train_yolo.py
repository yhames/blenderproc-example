"""
YOLO 학습 스크립트
"""

from ultralytics import YOLO

# 모델 로드
model = YOLO('yolo11n.pt')

# 학습 시작
results = model.train(
    data='dataset/yolo/data.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    device='cpu',
    project='runs/detect',
    name='train',
    exist_ok=True,
    patience=20,
    save=True,
    plots=True,
    verbose=True,
    
    # 최적화 설정
    optimizer='auto',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    
    # 증강 설정
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
)

print("\n" + "="*60)
print("학습 완료!")
print("="*60)
print(f"가중치 저장 위치: runs/detect/train/weights/best.pt")
print("="*60)
