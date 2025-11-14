#!/usr/bin/env python3
"""
BlenderProc 고급 실습 - YOLO 학습용 데이터셋 구축
BOP 객체와 다중 뷰포인트를 활용한 합성 데이터 생성
"""

import blenderproc as bproc
import numpy as np
import argparse
import os
import json
from pathlib import Path

# 커맨드 라인 인자
parser = argparse.ArgumentParser(description='BlenderProc YOLO 데이터셋 생성')
parser.add_argument('--num_scenes', type=int, default=10, help='생성할 씬 수')
parser.add_argument('--num_cameras_per_scene', type=int, default=10, help='씬당 카메라 포즈 수')
parser.add_argument('--output_dir', type=str, default='output_yolo', help='출력 디렉토리')
args = parser.parse_args()

print("=" * 50)
print("BlenderProc YOLO 데이터셋 생성")
print("로봇 비전 데이터셋 구축")
print("=" * 50)
print(f"씬 수: {args.num_scenes}")
print(f"씬당 카메라 포즈: {args.num_cameras_per_scene}")
print(f"총 이미지 수: {args.num_scenes * args.num_cameras_per_scene}")
print()

# ====================================
# 1. BlenderProc 초기화
# ====================================
print("[Step 1] BlenderProc 초기화 중...")
bproc.init()
print("✓ BlenderProc 초기화 완료")

# ====================================
# 2. 출력 디렉토리 설정
# ====================================
output_dir = Path(args.output_dir)
images_dir = output_dir / "images"
labels_dir = output_dir / "labels"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# YOLO 클래스 정의 (BOP YCB-V 객체)
class_names = [
    "002_master_chef_can",
    "003_cracker_box", 
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana"
]

# classes.txt 저장
with open(output_dir / "classes.txt", "w") as f:
    for name in class_names:
        f.write(f"{name}\n")

print(f"✓ 출력 디렉토리: {output_dir}")
print(f"✓ YOLO 클래스 수: {len(class_names)}")

# ====================================
# 3. BOP 데이터셋 경로 확인
# ====================================
print("\n[Step 2] BOP YCB-V 데이터셋 확인 중...")

# BOP YCB-V 데이터셋 경로 (blenderproc download ycbv로 다운로드된 경로)
bop_parent_path = os.path.join(os.path.dirname(__file__), "resources", "bop")
bop_dataset_name = "ycbv"
bop_dataset_path = os.path.join(bop_parent_path, bop_dataset_name)

if not os.path.exists(bop_dataset_path):
    print(f"⚠️  BOP YCB-V 데이터셋을 찾을 수 없습니다: {bop_dataset_path}")
    print("다음 명령으로 다운로드하세요: blenderproc download ycbv")
    exit(1)

print(f"✓ BOP 데이터셋 경로: {bop_dataset_path}")

# ====================================
# 4. 환경 구성 (테이블과 배경)
# ====================================
print("\n[Step 3] 환경 구성 중...")

# 지면 (바닥)
ground = bproc.object.create_primitive('PLANE', scale=[5, 5, 1])
ground.set_location([0, 0, 0])
ground.set_cp("category_id", 0)  # 배경

# 테이블 (작업 공간)
table = bproc.object.create_primitive('CUBE', scale=[0.8, 0.8, 0.05])
table.set_location([0, 0, 0.35])
table.set_cp("category_id", 0)  # 배경

print("✓ 환경 구성 완료 (지면, 테이블)")

# ====================================
# 5. 조명 설정
# ====================================
print("\n[Step 4] 조명 시스템 구성 중...")

# Key Light (주 조명)
light_key = bproc.types.Light()
light_key.set_type("SUN")
light_key.set_location([2, -2, 3])
light_key.set_energy(1.5)

# Fill Light (보조 조명)
light_fill = bproc.types.Light()
light_fill.set_type("SUN") 
light_fill.set_location([-2, 2, 2])
light_fill.set_energy(0.8)

# Rim Light (윤곽 조명)
light_rim = bproc.types.Light()
light_rim.set_type("POINT")
light_rim.set_location([0, 0, 2])
light_rim.set_energy(300)

print("✓ 3-point 조명 시스템 완료")

# ====================================
# 6. 카메라 설정
# ====================================
print("\n[Step 5] 카메라 설정 중...")

# 카메라 해상도 설정 (YOLO 입력 크기)
bproc.camera.set_resolution(640, 640)

# BOP 데이터셋의 intrinsics 로드
bproc.loader.load_bop_intrinsics(bop_dataset_path=bop_dataset_path)

print("✓ 카메라 설정 완료 (640x640)")

# ====================================
# 7. 렌더링 설정
# ====================================
print("\n[Step 6] 렌더링 설정 중...")

# RGB, Depth, Segmentation 활성화
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(
    map_by=["category_id", "instance", "name"],
    default_values={"category_id": 0}
)

# 렌더링 품질 설정
bproc.renderer.set_max_amount_of_samples(50)

print("✓ 렌더링 설정 완료")

# ====================================
# 8. 데이터 생성 루프
# ====================================
print(f"\n{'='*50}")
print("[YOLO 데이터셋 생성 시작]")
print("="*50)

# BOP 객체 ID (YCB-V에서 사용할 객체들)
obj_ids_to_use = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 10개 객체

frame_id = 0

for scene_id in range(args.num_scenes):
    print(f"\n[Scene {scene_id + 1}/{args.num_scenes}]")
    
    # ====================================
    # 8-1. BOP 객체 로드 (씬마다 다르게)
    # ====================================
    print("  - BOP 객체 로드 중...")
    
    # 이전 씬의 객체 제거
    if scene_id > 0:
        for obj in bop_objs:
            obj.delete()
    
    # 랜덤하게 3-6개 객체 선택
    num_objects = np.random.randint(3, 7)
    selected_obj_ids = np.random.choice(obj_ids_to_use, size=num_objects, replace=False)
    
    # BOP 객체 로드
    bop_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=bop_dataset_path,
        mm2m=True,
        obj_ids=selected_obj_ids.tolist()
    )
    
    # 각 객체에 category_id 설정 (YOLO 클래스)
    for obj in bop_objs:
        obj_id = obj.get_cp("bop_dataset_name")
        # BOP obj_id에서 YOLO class_id로 매핑
        class_id = obj.get_cp("category_id")
        obj.set_cp("category_id", class_id)
        obj.set_shading_mode('auto')
    
    print(f"    ✓ {len(bop_objs)}개 객체 로드 완료")
    
    # ====================================
    # 8-2. 객체 위치 샘플링 (도메인 랜덤화)
    # ====================================
    print("  - 객체 위치 랜덤화 중...")
    
    # 테이블 위에 객체 배치 함수
    def sample_pose_on_table(obj: bproc.types.MeshObject):
        # 테이블 위 랜덤 위치 (x: -0.3~0.3, y: -0.3~0.3, z: 0.4~0.6)
        obj.set_location(np.random.uniform([-0.3, -0.3, 0.42], [0.3, 0.3, 0.6]))
        # 랜덤 회전
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
    # 충돌 검사와 함께 객체 위치 샘플링
    bproc.object.sample_poses(
        objects_to_sample=bop_objs,
        sample_pose_func=sample_pose_on_table,
        max_tries=1000
    )
    
    print("    ✓ 객체 위치 샘플링 완료")
    
    # ====================================
    # 8-3. 조명 랜덤화
    # ====================================
    print("  - 조명 랜덤화 중...")
    
    # Key Light 에너지 랜덤화
    light_key.set_energy(np.random.uniform(1.0, 2.5))
    
    # Fill Light 에너지 랜덤화
    light_fill.set_energy(np.random.uniform(0.5, 1.5))
    
    # Point Light 위치와 에너지 랜덤화
    light_rim.set_location(bproc.sampler.shell(
        center=[0, 0, 0.5],
        radius_min=1.0,
        radius_max=2.0,
        elevation_min=30,
        elevation_max=89
    ))
    light_rim.set_energy(np.random.uniform(200, 500))
    
    print("    ✓ 조명 랜덤화 완료")
    
    # ====================================
    # 8-4. 배경 색상 랜덤화
    # ====================================
    print("  - 배경 랜덤화 중...")
    
    # 테이블 재질 랜덤화
    table_material = table.get_materials()[0]
    table_material.set_principled_shader_value(
        "Base Color", 
        np.random.uniform([0.3, 0.3, 0.3, 1], [0.8, 0.8, 0.8, 1])
    )
    
    # 바닥 재질 랜덤화
    ground_material = ground.get_materials()[0]
    ground_material.set_principled_shader_value(
        "Base Color",
        np.random.uniform([0.2, 0.2, 0.2, 1], [0.6, 0.6, 0.6, 1])
    )
    
    print("    ✓ 배경 랜덤화 완료")
    
    # ====================================
    # 8-5. 카메라 포즈 샘플링
    # ====================================
    print(f"  - {args.num_cameras_per_scene}개 카메라 포즈 샘플링 중...")
    
    # BVH 트리 생성 (카메라 장애물 검사용)
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(bop_objs)
    
    camera_poses = 0
    attempts = 0
    max_attempts = args.num_cameras_per_scene * 100
    
    while camera_poses < args.num_cameras_per_scene and attempts < max_attempts:
        attempts += 1
        
        # 구면 좌표계로 카메라 위치 샘플링
        location = bproc.sampler.shell(
            center=[0, 0, 0.5],
            radius_min=0.8,
            radius_max=1.5,
            elevation_min=20,
            elevation_max=80,
            uniform_volume=False
        )
        
        # POI (Point of Interest) 계산: 객체들의 중심
        poi = bproc.object.compute_poi(bop_objs)
        
        # 카메라가 POI를 바라보도록 회전 행렬 계산
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - location,
            inplane_rot=np.random.uniform(-0.7854, 0.7854)  # ±45도
        )
        
        # 카메라 포즈 행렬 생성
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # 장애물 검사: 카메라와 객체 사이에 최소 0.3m 거리 확보
        if bproc.camera.perform_obstacle_in_view_check(
            cam2world_matrix, 
            {"min": 0.3}, 
            bop_bvh_tree
        ):
            bproc.camera.add_camera_pose(cam2world_matrix)
            camera_poses += 1
    
    if camera_poses < args.num_cameras_per_scene:
        print(f"    ⚠️  경고: {camera_poses}/{args.num_cameras_per_scene}개 카메라 포즈만 생성됨")
    else:
        print(f"    ✓ {camera_poses}개 카메라 포즈 샘플링 완료")
    
    # ====================================
    # 8-6. 렌더링
    # ====================================
    print("  - 렌더링 중...")
    data = bproc.renderer.render()
    
    # ====================================
    # 8-7. YOLO 형식으로 저장
    # ====================================
    print("  - YOLO 형식으로 저장 중...")
    
    # 각 카메라 프레임에 대해
    for cam_idx in range(len(data["colors"])):
        # RGB 이미지 저장
        img_filename = f"{frame_id:06d}.png"
        img_path = images_dir / img_filename
        
        import cv2
        cv2.imwrite(str(img_path), data["colors"][cam_idx][..., ::-1])  # RGB to BGR
        
        # YOLO 라벨 생성
        label_filename = f"{frame_id:06d}.txt"
        label_path = labels_dir / label_filename
        
        # Instance segmentation에서 객체별 바운딩 박스 추출
        instance_segmap = data["instance_segmaps"][cam_idx]
        instance_attribute_map = data["instance_attribute_maps"][cam_idx]
        
        # 이미지 크기
        h, w = instance_segmap.shape
        
        yolo_labels = []
        
        # 각 인스턴스에 대해
        for instance_id in np.unique(instance_segmap):
            if instance_id == 0:  # 배경 제외
                continue
            
            # 해당 인스턴스의 마스크
            mask = (instance_segmap == instance_id)
            
            # category_id 가져오기
            if instance_id < len(instance_attribute_map):
                category_id = instance_attribute_map[instance_id]["category_id"]
                
                # 배경(category_id=0) 제외
                if category_id == 0:
                    continue
                
                # YOLO class_id (0-based index)
                class_id = category_id - 1  # BOP category_id는 1부터 시작
                
                if class_id < 0 or class_id >= len(class_names):
                    continue
                
                # 바운딩 박스 계산
                y_indices, x_indices = np.where(mask)
                
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue
                
                x_min = np.min(x_indices)
                x_max = np.max(x_indices)
                y_min = np.min(y_indices)
                y_max = np.max(y_indices)
                
                # YOLO 형식으로 변환 (중심 x, 중심 y, 너비, 높이, 정규화)
                x_center = (x_min + x_max) / 2.0 / w
                y_center = (y_min + y_max) / 2.0 / h
                bbox_width = (x_max - x_min) / w
                bbox_height = (y_max - y_min) / h
                
                # YOLO 라벨 추가
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        # 라벨 파일 저장
        with open(label_path, "w") as f:
            for label in yolo_labels:
                f.write(label + "\n")
        
        print(f"    ✓ Frame {frame_id}: {len(yolo_labels)}개 객체 라벨링")
        frame_id += 1

print(f"\n{'='*50}")
print("[데이터 생성 완료]")
print("="*50)

print(f"✓ 총 생성된 이미지: {frame_id}개")
print(f"✓ 저장 위치: {output_dir}")
print(f"✓ 이미지: {images_dir}")
print(f"✓ 라벨: {labels_dir}")

# ====================================
# 9. 데이터셋 분할 (train/val)
# ====================================
print("\n[데이터셋 분할 생성 중...]")

# train/val 분할 (80:20)
image_files = sorted(images_dir.glob("*.png"))
num_images = len(image_files)
num_train = int(num_images * 0.8)

train_files = image_files[:num_train]
val_files = image_files[num_train:]

# train.txt 생성
with open(output_dir / "train.txt", "w") as f:
    for img_file in train_files:
        f.write(f"{img_file}\n")

# val.txt 생성
with open(output_dir / "val.txt", "w") as f:
    for img_file in val_files:
        f.write(f"{img_file}\n")

print(f"✓ Train: {len(train_files)}개")
print(f"✓ Val: {len(val_files)}개")

# ====================================
# 10. YOLO 설정 파일 생성
# ====================================
print("\n[YOLO 설정 파일 생성 중...]")

yaml_content = f"""# YOLO Dataset Configuration
path: {output_dir.absolute()}
train: train.txt
val: val.txt

# Classes
nc: {len(class_names)}
names: {class_names}
"""

with open(output_dir / "dataset.yaml", "w") as f:
    f.write(yaml_content)

print(f"✓ dataset.yaml 생성 완료")

# ====================================
# 11. 학습 포인트 요약
# ====================================
print(f"\n{'='*50}")
print("[학습 포인트 요약]")
print("="*50)

print(f"""
✓ BlenderProc 주요 기능:
  1. BOP 데이터셋 로더 활용 (YCB-V)
  2. 도메인 랜덤화 (객체 위치, 회전, 조명, 배경)
  3. 다중 카메라 뷰포인트 샘플링
  4. Instance Segmentation 기반 바운딩 박스 추출
  5. YOLO 형식 데이터셋 생성

✓ 도메인 랜덤화 요소:
  - 객체 개수 (3-6개)
  - 객체 위치와 회전
  - 조명 강도와 위치
  - 배경 색상
  - 카메라 뷰포인트

✓ 생성된 데이터:
  - {frame_id}개 RGB 이미지 (640x640)
  - YOLO 형식 바운딩 박스 라벨
  - Train/Val 분할 (80:20)
  - {len(class_names)}개 클래스

✓ YOLO 학습 방법:
  1. YOLOv8 설치: pip install ultralytics
  2. 학습 실행:
     from ultralytics import YOLO
     model = YOLO('yolov8n.pt')
     model.train(data='{output_dir / "dataset.yaml"}', epochs=100)

✓ Isaac Sim 대비 BlenderProc 장점:
  - 가볍고 빠른 렌더링
  - 쉬운 설치와 사용
  - 다양한 3D 데이터셋 지원 (BOP, ShapeNet, etc.)
  - 오픈소스 & 무료
""")

print("\n✓ 프로그램 종료")
