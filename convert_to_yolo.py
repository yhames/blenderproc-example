"""
Step 2: HDF5 → YOLO 형식 변환
Instance Segmentation에서 Bounding Box 추출 후 YOLO 형식으로 저장
"""

import h5py
import numpy as np
import cv2
import os
import glob
from pathlib import Path

# 카테고리 매핑 (BlenderProc category_id → YOLO class_id)
CATEGORY_TO_CLASS = {
    1: 0,  # PottedMeatCan → class 0
    2: 1,  # Banana → class 1
    3: 2,  # LargeMarker → class 2
    4: 3,  # TomatoSoupCan → class 3
}

CLASS_NAMES = ['meat_can', 'banana', 'marker', 'soup_can']


def extract_bbox_from_mask(mask):
    """
    마스크에서 바운딩 박스 추출
    Returns: [x_min, y_min, x_max, y_max] or None
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # 가장 큰 contour 선택
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return [x, y, x + w, y + h]


def bbox_to_yolo(bbox, img_width, img_height):
    """
    [x_min, y_min, x_max, y_max] → YOLO [x_center, y_center, width, height]
    모든 값 0~1로 정규화
    """
    x_min, y_min, x_max, y_max = bbox
    
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return [x_center, y_center, width, height]


def process_hdf5_to_yolo(hdf5_path, output_base_dir, scene_name, camera_idx):
    """
    HDF5 파일 → YOLO 형식 변환
    """
    with h5py.File(hdf5_path, 'r') as f:
        # 데이터 로드
        colors = f['colors'][:]  # RGB (H, W, 3)
        instance_segmaps = f['instance_segmaps'][:]  # (H, W)
        
        # Category segmentation 로드
        if 'category_id_segmaps' in f:
            category_segmaps = f['category_id_segmaps'][:]
        else:
            category_segmaps = None
        
        img_height, img_width = colors.shape[:2]
        
        # 파일명 생성
        image_filename = f"{scene_name}_cam{camera_idx}.png"
        label_filename = f"{scene_name}_cam{camera_idx}.txt"
        
        # 이미지 저장
        image_path = output_base_dir / "images" / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))
        
        # Instance별 bbox 추출
        unique_instances = np.unique(instance_segmaps)
        unique_instances = unique_instances[unique_instances > 0]  # 배경 제외
        
        # YOLO 라벨 생성
        yolo_labels = []
        
        for inst_id in unique_instances:
            # Instance 마스크
            mask = (instance_segmaps == inst_id).astype(np.uint8)
            
            # Category ID 가져오기
            if category_segmaps is not None:
                category_id = int(np.median(category_segmaps[mask > 0]))
            else:
                category_id = inst_id  # fallback
            
            # YOLO class_id로 변환
            if category_id not in CATEGORY_TO_CLASS:
                continue
            
            class_id = CATEGORY_TO_CLASS[category_id]
            
            # Bbox 추출
            bbox = extract_bbox_from_mask(mask)
            if bbox is None:
                continue
            
            # YOLO 형식으로 변환
            yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
            
            # 라벨 추가
            yolo_labels.append([class_id] + yolo_bbox)
        
        # 라벨 파일 저장
        label_path = output_base_dir / "labels" / label_filename
        with open(label_path, 'w') as f:
            for label in yolo_labels:
                class_id, x_c, y_c, w, h = label
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        
        return len(yolo_labels)


def convert_all_hdf5_to_yolo(input_dir="dataset/raw", output_dir="dataset/yolo"):
    """
    모든 HDF5 파일을 YOLO 형식으로 변환
    """
    print("=" * 60)
    print("HDF5 → YOLO 형식 변환")
    print("=" * 60)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 생성
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)
    
    # 모든 HDF5 파일 찾기
    hdf5_files = list(input_path.glob("scene_*/[0-9].hdf5"))
    
    if not hdf5_files:
        print(f"✗ HDF5 파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"\n발견된 HDF5 파일: {len(hdf5_files)}개")
    print(f"출력 디렉토리: {output_path}\n")
    
    total_images = 0
    total_objects = 0
    
    # 각 HDF5 파일 처리
    for hdf5_file in sorted(hdf5_files):
        scene_name = hdf5_file.parent.name  # scene_0000
        camera_idx = hdf5_file.stem  # 0, 1, 2
        
        num_objects = process_hdf5_to_yolo(
            hdf5_file, 
            output_path, 
            scene_name, 
            camera_idx
        )
        
        total_images += 1
        total_objects += num_objects
        
        print(f"✓ {scene_name}/cam{camera_idx}: {num_objects}개 객체")
    
    print(f"\n{'='*60}")
    print("변환 완료!")
    print("=" * 60)
    print(f"총 이미지: {total_images}개")
    print(f"총 객체: {total_objects}개")
    print(f"평균 객체/이미지: {total_objects/total_images:.1f}개\n")
    
    # data.yaml 생성
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_path.absolute()}
train: images
val: images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ data.yaml 생성: {yaml_path}")
    
    print(f"\n디렉토리 구조:")
    print(f"  {output_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── scene_0000_cam0.png")
    print(f"    │   ├── scene_0000_cam1.png")
    print(f"    │   └── ...")
    print(f"    ├── labels/")
    print(f"    │   ├── scene_0000_cam0.txt")
    print(f"    │   ├── scene_0000_cam1.txt")
    print(f"    │   └── ...")
    print(f"    └── data.yaml")
    
    print(f"\n다음 단계:")
    print(f"  1. Train/Val 분리 (선택)")
    print(f"  2. YOLO 학습:")
    print(f"     yolo task=detect mode=train model=yolov8n.pt data={yaml_path} epochs=100")
    print("=" * 60)


if __name__ == "__main__":
    convert_all_hdf5_to_yolo()
