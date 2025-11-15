import bpy
import os

# ======================================================
# 사용자 설정
# ======================================================
# 현재 스크립트 위치 기준으로 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_FOLDER = os.path.join(SCRIPT_DIR, "assets", "ycb_usd")      # USD 파일들이 있는 폴더
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "assets", "ycb_obj")   # OBJ로 변환될 폴더

print(f"[INFO] USD 폴더: {USD_FOLDER}")
print(f"[INFO] 출력 폴더: {OUTPUT_FOLDER}")

# 폴더 존재 확인
if not os.path.exists(USD_FOLDER):
    print(f"[ERROR] USD 폴더를 찾을 수 없습니다: {USD_FOLDER}")
    exit(1)

# 출력 폴더 없으면 생성
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================================================
# 변환 함수
# ======================================================
def convert_usd_to_obj(usd_path, output_path):
    # 현재 장면 초기화
    bpy.ops.wm.read_factory_settings(use_empty=True)

    print(f"[INFO] Loading USD: {usd_path}")
    bpy.ops.wm.usd_import(filepath=usd_path)

    print(f"[INFO] Exporting OBJ: {output_path}")
    
    # Blender 4.2+ 에서는 wm.obj_export 사용
    try:
        bpy.ops.wm.obj_export(
            filepath=output_path,
            export_selected_objects=False,
            export_materials=True,
            export_triangulated_mesh=False,
            apply_modifiers=True,
            export_normals=True,
            export_uv=True
        )
    except AttributeError:
        # 구버전 Blender용 fallback
        bpy.ops.export_scene.obj(filepath=output_path, use_selection=False)

    print(f"[DONE] {usd_path} → {output_path}")


# ======================================================
# USD 파일 전체 변환
# ======================================================
usd_files = [f for f in os.listdir(USD_FOLDER) if f.endswith(".usd")]

if not usd_files:
    print("[ERROR] USD 파일을 찾을 수 없습니다.")
else:
    print(f"[INFO] 총 {len(usd_files)}개의 USD 변환 시작")

for usd_file in usd_files:
    usd_path = os.path.join(USD_FOLDER, usd_file)
    obj_path = os.path.join(OUTPUT_FOLDER, usd_file.replace(".usd", ".obj"))

    convert_usd_to_obj(usd_path, obj_path)

print("==========================================")
print("✓ 모든 USD → OBJ 변환 완료")
print("==========================================")
