import blenderproc as bproc

import argparse
import os

import numpy as np

# 커맨드 라인 인자
parser = argparse.ArgumentParser(description='BlenderProc 데이터셋 생성')
parser.add_argument('--num_scenes', type=int, default=5, help='생성할 씬 수')
parser.add_argument('--output_dir', type=str, default='dataset/raw', help='출력 디렉토리')
args = parser.parse_args()

print("=" * 60)
print("BlenderProc 데이터셋 생성")
print("=" * 60)
print(f"씬 수: {args.num_scenes}")
print(f"출력: {args.output_dir}")
print()

# BlenderProc 초기화
bproc.init()

# 출력 디렉토리
output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# ====================================
# 조명 설정
# ====================================
print("[1/6] 조명 설정...")
key_light = bproc.types.Light()
key_light.set_type("SUN")
key_light.set_location([2, -2, 3])
key_light.set_rotation_euler([-0.785, 0, -0.785])
key_light.set_energy(2.0)

fill_light = bproc.types.Light()
fill_light.set_type("SUN")
fill_light.set_location([-2, 2, 2.5])
fill_light.set_rotation_euler([-0.611, 0, 0.785])
fill_light.set_energy(0.8)

# ====================================
# 환경 구성
# ====================================
print("[2/6] 환경 구성...")
ground = bproc.object.create_primitive('PLANE', scale=[10, 10, 1], location=[0, 0, 0])
ground.set_name("Ground")
ground.set_cp("category_id", 0)

ground_mat = bproc.material.create('GroundMat')
ground_mat.set_principled_shader_value("Base Color", [0.8, 0.8, 0.8, 1.0])
ground.replace_materials(ground_mat)

table = bproc.object.create_primitive('CUBE', scale=[0.8, 0.8, 0.05], location=[0.5, 0.0, 0.35])
table.set_name("Table")
table.set_cp("category_id", 0)

table_mat = bproc.material.create('TableMat')
table_mat.set_principled_shader_value("Base Color", [0.5, 0.5, 0.5, 1.0])
table.replace_materials(table_mat)

table.enable_rigidbody(False)
ground.enable_rigidbody(False)

# ====================================
# YCB 객체 로드
# ====================================
print("[3/6] YCB 객체 로드...")
ycb_dir = os.path.join(os.path.dirname(__file__), "assets", "ycb_obj")

ycb_objects_info = [
    {"name": "PottedMeatCan", "file": "010_potted_meat_can.obj", "category_id": 1, "color": [0.8, 0.6, 0.4, 1.0]},
    {"name": "Banana", "file": "011_banana.obj", "category_id": 2, "color": [0.9, 0.8, 0.2, 1.0]},
    {"name": "LargeMarker", "file": "040_large_marker.obj", "category_id": 3, "color": [0.2, 0.3, 0.8, 1.0]},
    {"name": "TomatoSoupCan", "file": "005_tomato_soup_can.obj", "category_id": 4, "color": [0.9, 0.2, 0.2, 1.0]},
]

ycb_objects = []
for obj_info in ycb_objects_info:
    obj_path = os.path.join(ycb_dir, obj_info["file"])

    if os.path.exists(obj_path):
        # 재질 생성
        mat = bproc.material.create(f'{obj_info["name"]}_Mat')
        mat.set_principled_shader_value("Base Color", obj_info["color"])
        mat.set_principled_shader_value("Roughness", 0.5)

        # OBJ 로드
        loaded_objs = bproc.loader.load_obj(obj_path)

        for idx, obj in enumerate(loaded_objs):
            obj.set_name(f"{obj_info['name']}_{idx}" if len(loaded_objs) > 1 else obj_info["name"])
            obj.set_cp("category_id", obj_info["category_id"])
            obj.clear_materials()
            obj.add_material(mat)
            obj.enable_rigidbody(True, mass=0.1, friction=1.0, linear_damping=0.99, angular_damping=0.99)
            ycb_objects.append(obj)

        print(f"  ✓ {obj_info['name']}: {len(loaded_objs)}개 메쉬")

print(f"✓ 총 {len(ycb_objects)}개 객체 로드 완료")

# ====================================
# 렌더링 설정
# ====================================
print("[4/6] 렌더링 설정...")
bproc.renderer.set_max_amount_of_samples(128)
bproc.renderer.enable_segmentation_output(
    map_by=["category_id", "instance"],
    default_values={"category_id": 0}
)
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# ====================================
# 씬 생성 루프
# ====================================
print(f"\n[5/6] {args.num_scenes}개 씬 생성 & 렌더링 중...")

for scene_idx in range(args.num_scenes):
    print(f"\n  Scene {scene_idx + 1}/{args.num_scenes}")

    # 카메라 포즈 초기화 (이전 씬의 카메라 제거)
    bproc.utility.reset_keyframes()

    # 객체 랜덤 배치
    for obj in ycb_objects:
        random_pos = np.random.uniform([0.3, -0.25, 0.42], [0.7, 0.25, 0.6])
        obj.set_location(random_pos)
        random_rot = np.random.uniform([0, 0, 0], [360, 360, 360])
        obj.set_rotation_euler([np.deg2rad(r) for r in random_rot])

    # 물리 시뮬레이션
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=0.5,
        max_simulation_time=1.0,
        check_object_interval=0.25
    )

    # 조명 랜덤화
    key_light.set_energy(np.random.uniform(1.5, 3.0))
    fill_light.set_energy(np.random.uniform(0.5, 1.5))

    # 테이블 색상 랜덤화
    random_color = np.random.uniform([0.3, 0.3, 0.3], [0.8, 0.8, 0.8])
    table_mat.set_principled_shader_value("Base Color", [*random_color, 1.0])

    # 다중 카메라 포즈 (3개 뷰)
    poi = np.array([0.5, 0.0, 0.45])  # Point of Interest (테이블 중심)

    # 1. 메인 카메라 - 랜덤 위치
    cam_position = np.random.uniform([0.8, 0.8, 0.6], [1.4, 1.4, 1.2])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        poi - cam_position,
        inplane_rot=np.random.uniform(-0.2, 0.2)
    )
    cam_matrix = bproc.math.build_transformation_mat(cam_position, rotation_matrix)
    bproc.camera.add_camera_pose(cam_matrix)

    # 2. 탑뷰 카메라 - 위에서 내려다봄 (고정)
    cam_position = np.array([0.5, 0.0, 1.5])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_position)
    cam_matrix = bproc.math.build_transformation_mat(cam_position, rotation_matrix)
    bproc.camera.add_camera_pose(cam_matrix)

    # 3. 사이드 카메라 - 옆에서 바라봄 (고정)
    cam_position = np.array([1.5, 0.0, 0.6])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_position)
    cam_matrix = bproc.math.build_transformation_mat(cam_position, rotation_matrix)
    bproc.camera.add_camera_pose(cam_matrix)

    # 렌더링
    data = bproc.renderer.render()

    # 개별 HDF5 파일로 저장
    scene_output_dir = os.path.join(output_dir, f"scene_{scene_idx:04d}")
    os.makedirs(scene_output_dir, exist_ok=True)

    bproc.writer.write_hdf5(
        scene_output_dir,
        data,
        append_to_existing_output=False
    )

    print(f"    ✓ 렌더링 & 저장 완료: scene_{scene_idx:04d}/ (3개 카메라 뷰)")

print(f"\n✓ 모든 씬 생성 완료")

# ====================================
# 결과 요약
# ====================================
print("\n" + "=" * 60)
print("✓ 데이터 생성 완료!")
print("=" * 60)
print(f"출력 디렉토리: {output_dir}")
print(f"생성된 씬: {args.num_scenes}개")
print(f"\n각 씬마다 다음 데이터가 저장됨:")
print(f"  - RGB 이미지 (3개 카메라)")
print(f"  - Instance Segmentation (3개 카메라)")
print(f"  - Depth Map (3개 카메라)")
print(f"  - Category ID")
print(f"\n디렉토리 구조:")
print(f"  {args.output_dir}/")
print(f"    ├── scene_0000/")
print(f"    │   ├── 0.hdf5 (메인 카메라)")
print(f"    │   ├── 1.hdf5 (탑뷰 카메라)")
print(f"    │   └── 2.hdf5 (사이드 카메라)")
print(f"    ├── scene_0001/")
print(f"    └── ...")
print("\n다음 단계: python convert_to_yolo.py")
print("=" * 60)
