import blenderproc as bproc

import numpy as np
import argparse
import os
import time
import glob

# 커맨드 라인 인자
parser = argparse.ArgumentParser(description='BlenderProc 고급 실습')
parser.add_argument('--num_scenes', type=int, default=10, help='생성할 씬 수')
parser.add_argument('--num_cameras_per_scene', type=int, default=3, help='씬당 카메라 뷰 수')
parser.add_argument('--resolution', type=int, default=512, help='렌더링 해상도')
args = parser.parse_args()

print("=" * 50)
print("BlenderProc 고급 실습")
print("로봇 비전 데이터셋 구축")
print("=" * 50)
print(f"씬 수: {args.num_scenes}")
print(f"씬당 카메라: {args.num_cameras_per_scene}")
print(f"해상도: {args.resolution}x{args.resolution}")
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
output_dir = os.path.join(os.path.dirname(__file__), "blenderproc_output", "advanced_dataset")
os.makedirs(output_dir, exist_ok=True)
print(f"✓ 출력 디렉토리: {output_dir}")

# ====================================
# 3. 조명 설정 (고급 3-point lighting)
# ====================================
print("[Step 2] 고급 조명 시스템 구성 중...")

# Key Light (주 조명) - 강한 방향성 조명
key_light = bproc.types.Light()
key_light.set_type("SUN")
key_light.set_location([2, -2, 3])
key_light.set_rotation_euler([-0.785, 0, -0.785])  # -45도, 0도, -45도 (라디안)
key_light.set_energy(2.0)

# Fill Light (보조 조명) - 부드러운 보조광
fill_light = bproc.types.Light()
fill_light.set_type("SUN")
fill_light.set_location([-2, 2, 2.5])
fill_light.set_rotation_euler([-0.611, 0, 0.785])  # -35도, 0도, 45도
fill_light.set_energy(0.8)

# Ambient Light (환경광) - HDRI 대신 반구 조명
ambient_light = bproc.types.Light()
ambient_light.set_type("AREA")
ambient_light.set_location([0, 0, 5])
ambient_light.set_rotation_euler([0, 0, 0])
ambient_light.set_energy(100)
ambient_light.blender_obj.data.size = 10

print("✓ 3-point 조명 시스템 완료")

# ====================================
# 4. 환경 구성
# ====================================
print("[Step 3] 환경 구성 중...")

# 지면 (plane)
ground = bproc.object.create_primitive('PLANE', scale=[10, 10, 1], location=[0, 0, 0])
ground.set_name("GroundPlane")
ground.set_cp("category_id", 0)  # 배경으로 설정

# 지면 재질 설정
ground_mat = bproc.material.create('GroundMaterial')
ground_mat.set_principled_shader_value("Base Color", [0.8, 0.8, 0.8, 1.0])
ground_mat.set_principled_shader_value("Roughness", 0.8)
ground.replace_materials(ground_mat)

# 테이블 (작업 공간)
table = bproc.object.create_primitive(
    'CUBE',
    scale=[0.8, 0.8, 0.05],
    location=[0.5, 0.0, 0.35]
)
table.set_name("Table")
table.set_cp("category_id", 0)  # 배경으로 설정

# 테이블 재질 설정
table_mat = bproc.material.create('TableMaterial')
table_mat.set_principled_shader_value("Base Color", [0.5, 0.5, 0.5, 1.0])
table_mat.set_principled_shader_value("Roughness", 0.6)
table.replace_materials(table_mat)

# 테이블과 지면을 움직이지 않도록 설정
table.enable_rigidbody(False)
ground.enable_rigidbody(False)

print("✓ 환경 구성 완료")

# ====================================
# 5. YCB 객체 로드
# ====================================
print("[Step 4] YCB 객체 로드 중...")

# YCB OBJ 파일 경로
ycb_dir = os.path.join(os.path.dirname(__file__), "ycb_obj")

# YCB 객체 정의
ycb_objects_info = [
    {
        "name": "PottedMeatCan",
        "file": "010_potted_meat_can.obj",
        "position": [0.5, 0.0, 0.42],
        "rotation": [90, 0, 0],  # X축 90도 회전 (OBJ는 다른 축 방향)
        "scale": [1.0, 1.0, 1.0],
        "category_id": 1
    },
    {
        "name": "Banana",
        "file": "011_banana.obj",
        "position": [0.4, 0.15, 0.42],
        "rotation": [0, 0, 0],
        "scale": [1.0, 1.0, 1.0],
        "category_id": 2
    },
    {
        "name": "LargeMarker",
        "file": "040_large_marker.obj",
        "position": [0.6, -0.15, 0.42],
        "rotation": [0, 0, 0],
        "scale": [1.0, 1.0, 1.0],
        "category_id": 3
    },
    {
        "name": "TomatoSoupCan",
        "file": "005_tomato_soup_can.obj",
        "position": [0.35, -0.1, 0.42],
        "rotation": [0, 0, 0],
        "scale": [1.0, 1.0, 1.0],
        "category_id": 4
    }
]

# YCB 객체 로드
ycb_objects = []
for obj_info in ycb_objects_info:
    obj_path = os.path.join(ycb_dir, obj_info["file"])
    
    if os.path.exists(obj_path):
        # OBJ 파일 로드
        loaded_objs = bproc.loader.load_obj(obj_path)
        
        for obj in loaded_objs:
            obj.set_name(obj_info["name"])
            obj.set_location(obj_info["position"])
            obj.set_rotation_euler([np.deg2rad(r) for r in obj_info["rotation"]])
            obj.set_scale(obj_info["scale"])
            
            # 카테고리 ID 설정 (세그멘테이션용)
            obj.set_cp("category_id", obj_info["category_id"])
            
            # 물리 속성 활성화
            obj.enable_rigidbody(True, mass=0.1, friction=1.0, linear_damping=0.99, angular_damping=0.99)
            
            ycb_objects.append(obj)
            
        print(f"  ✓ {obj_info['name']} 로드 및 설정 완료")
    else:
        print(f"  ✗ {obj_info['name']} 파일을 찾을 수 없음: {obj_path}")

print(f"✓ YCB 객체 로드 완료 (총 {len(ycb_objects)}개)")

# ====================================
# 6. 물리 시뮬레이션 설정
# ====================================
print("[Step 5] 물리 시뮬레이션 설정 중...")

# 물리 시뮬레이션 설정
bproc.object.simulate_physics_and_fix_final_poses(
    min_simulation_time=0.5,
    max_simulation_time=2.0,
    check_object_interval=0.25
)

print("✓ 물리 시뮬레이션 설정 완료")

# ====================================
# 7. 렌더링 설정
# ====================================
print("[Step 6] 렌더링 설정 중...")

# 렌더링 품질 설정
bproc.renderer.set_max_amount_of_samples(128)
bproc.renderer.set_output_format(enable_transparency=False)
bproc.renderer.set_light_bounces(
    diffuse_bounces=3,
    glossy_bounces=3,
    max_bounces=3,
    transmission_bounces=3,
    transparent_max_bounces=3
)

# 세그멘테이션 및 깊이 출력 활성화
bproc.renderer.enable_segmentation_output(
    map_by=["category_id", "instance", "name"],
    default_values={"category_id": 0}
)
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_normals_output()

print("✓ 렌더링 설정 완료")

# ====================================
# 8. 데이터 생성 루프 (도메인 랜덤화)
# ====================================
print(f"\n{'='*50}")
print("[고급 데이터셋 생성 시작]")
print("="*50)
print(f"총 {args.num_scenes}개 씬 x {args.num_cameras_per_scene}개 카메라 = {args.num_scenes * args.num_cameras_per_scene}개 이미지")

start_time = time.time()

for scene_idx in range(args.num_scenes):
    print(f"\n[Scene {scene_idx + 1}/{args.num_scenes}]")
    
    # ================================
    # 도메인 랜덤화 적용
    # ================================
    
    # 1. YCB 객체 위치와 회전 랜덤화
    for obj in ycb_objects:
        # 테이블 위의 랜덤한 위치
        random_pos = np.random.uniform([0.3, -0.25, 0.42], [0.7, 0.25, 0.6])
        obj.set_location(random_pos)
        
        # 랜덤한 회전
        random_rot = np.random.uniform([0, 0, 0], [360, 360, 360])
        obj.set_rotation_euler([np.deg2rad(r) for r in random_rot])
    
    # 물리 시뮬레이션으로 자연스러운 배치
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=0.5,
        max_simulation_time=1.0,
        check_object_interval=0.25
    )
    
    # 2. 조명 강도 랜덤화
    key_light.set_energy(np.random.uniform(1.5, 3.0))
    fill_light.set_energy(np.random.uniform(0.5, 1.5))
    ambient_light.set_energy(np.random.uniform(50, 150))
    
    # 3. 조명 방향 랜덤화
    key_rotation = np.random.uniform([-0.873, -0.1, -0.873], [-0.698, 0.1, -0.698])
    key_light.set_rotation_euler(key_rotation)
    
    # 4. 테이블 색상 랜덤화
    random_table_color = np.random.uniform([0.3, 0.3, 0.3], [0.8, 0.8, 0.8])
    table_mat.set_principled_shader_value("Base Color", [*random_table_color, 1.0])
    
    # ================================
    # 다중 카메라 뷰포인트 생성
    # ================================
    
    for cam_idx in range(args.num_cameras_per_scene):
        
        if cam_idx == 0:
            # 메인 카메라 - 랜덤한 각도에서 테이블을 바라봄
            cam_position = np.random.uniform([0.8, 0.8, 0.6], [1.4, 1.4, 1.2])
            poi = np.array([0.5, 0.0, 0.45])  # Point of Interest (테이블 중심)
            
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                poi - cam_position,
                inplane_rot=np.random.uniform(-0.2, 0.2)
            )
            
            cam_matrix = bproc.math.build_transformation_mat(cam_position, rotation_matrix)
            bproc.camera.add_camera_pose(cam_matrix)
            
        elif cam_idx == 1:
            # 탑뷰 카메라 - 위에서 내려다봄
            cam_position = np.array([0.5, 0.0, 1.5])
            poi = np.array([0.5, 0.0, 0.45])
            
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_position)
            cam_matrix = bproc.math.build_transformation_mat(cam_position, rotation_matrix)
            bproc.camera.add_camera_pose(cam_matrix)
            
        else:
            # 사이드 카메라 - 옆에서 바라봄
            cam_position = np.array([1.5, 0.0, 0.6])
            poi = np.array([0.5, 0.0, 0.45])
            
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_position)
            cam_matrix = bproc.math.build_transformation_mat(cam_position, rotation_matrix)
            bproc.camera.add_camera_pose(cam_matrix)

print(f"\n✓ {args.num_scenes}개 씬, {args.num_scenes * args.num_cameras_per_scene}개 카메라 포즈 생성 완료")

# ====================================
# 9. 렌더링 실행
# ====================================
print(f"\n{'='*50}")
print("[렌더링 시작]")
print("="*50)

# 렌더링 실행
data = bproc.renderer.render()

# ====================================
# 10. 데이터 저장
# ====================================
print(f"\n[데이터 저장 중...]")

# HDF5 형식으로 저장
bproc.writer.write_hdf5(
    output_dir,
    data,
    append_to_existing_output=False
)

# COCO 형식으로 저장 (바운딩 박스 어노테이션)
bproc.writer.write_coco_annotations(
    os.path.join(output_dir, "coco_annotations"),
    instance_segmaps=data["instance_segmaps"],
    instance_attribute_maps=data["instance_attribute_maps"],
    colors=data["colors"],
    color_file_format="PNG",
    append_to_existing_output=False
)

end_time = time.time()
total_time = end_time - start_time

# ====================================
# 11. 결과 요약
# ====================================
print(f"\n{'='*50}")
print("[데이터 생성 완료]")
print("="*50)

total_images = args.num_scenes * args.num_cameras_per_scene
fps = total_images / total_time if total_time > 0 else 0

print(f"✓ 생성된 씬: {args.num_scenes}개")
print(f"✓ 생성된 이미지: {total_images}개")
print(f"✓ 소요 시간: {total_time:.2f}초")
print(f"✓ 평균 처리 속도: {fps:.2f} images/sec")
print(f"✓ 저장 위치: {output_dir}")

# 생성된 파일 통계
if os.path.exists(output_dir):
    hdf5_files = glob.glob(os.path.join(output_dir, "*.hdf5"))
    coco_dir = os.path.join(output_dir, "coco_annotations")
    
    print(f"\n[생성된 파일]")
    print(f"  - HDF5 파일: {len(hdf5_files)}개")
    
    if os.path.exists(coco_dir):
        coco_images = glob.glob(os.path.join(coco_dir, "*.png"))
        coco_json = glob.glob(os.path.join(coco_dir, "*.json"))
        print(f"  - COCO 이미지: {len(coco_images)}개")
        print(f"  - COCO JSON: {len(coco_json)}개")

# ====================================
# 12. 학습 포인트 요약
# ====================================
print(f"\n{'='*50}")
print("[BlenderProc 고급 학습 포인트 요약]")
print("="*50)

print(f"""
✓ 고급 BlenderProc 기능:
  1. 다중 카메라 시스템 ({args.num_cameras_per_scene}개 뷰포인트)
  2. YCB 객체 활용 (OBJ 형식)
  3. 물리 시뮬레이션 기반 배치
  4. 3-point 조명 시스템
  5. HDF5 + COCO 형식 출력

✓ 고급 도메인 랜덤화:
  - 객체 위치/회전 랜덤화
  - 물리 기반 자연스러운 배치
  - 조명 강도/방향 변화
  - 테이블 색상 변화
  - 카메라 궤도 랜덤화

✓ 생성된 데이터:
  - RGB 이미지
  - 깊이 맵 (Depth)
  - 법선 맵 (Normals)
  - 세그멘테이션 맵 (Category/Instance)
  - COCO 바운딩 박스 어노테이션
  - 카메라 파라미터

✓ Isaac Sim과의 차이점:
  - BlenderProc는 씬 단위로 작업
  - 물리 시뮬레이션은 렌더링 전 실행
  - COCO 형식 자동 변환 지원
  - HDF5로 통합 저장

✓ 실무 활용:
  - 로봇 그래스핑 학습
  - 객체 검출/세그멘테이션
  - 6-DoF 포즈 추정
  - Sim-to-Real 전이 학습
  - YOLOv8, Mask R-CNN 등 학습

✓ 다음 단계:
  - visHdf5Files.py로 결과 시각화
  - vis_coco_annotation.py로 바운딩 박스 확인
  - 더 많은 YCB 객체 추가
  - 텍스처 랜덤화 추가
  - 배경 이미지 합성
""")

print(f"\n{'='*50}")
print("✓ 프로그램 완료")
print("="*50)
