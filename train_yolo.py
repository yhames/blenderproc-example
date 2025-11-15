"""
YOLO 학습 스크립트
"""

from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')

# 학습 시작
results = model.train(
    data='dataset/yolo/data.yaml',
    epochs=10,
    batch=8,
    imgsz=512,
    device='cpu',
    project='runs/detect',
    name='train'
)

print("\n" + "="*60)
print("학습 완료!")
print("="*60)
print(f"가중치 저장 위치: runs/detect/train/weights/best.pt")
print("="*60)
