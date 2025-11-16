import os
import requests
import tarfile
from pathlib import Path

# ======================================================
# 사용자 설정
# ======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "assets", "ycb_objects")

# YCB 오브젝트 목록 (berkeley meshes)
YCB_OBJECTS = [
    "005_tomato_soup_can",
    "010_potted_meat_can",
    "011_banana",
    "040_large_marker",
]

BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/berkeley"

print(f"[INFO] 출력 폴더: {OUTPUT_FOLDER}")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ======================================================
# 다운로드 및 압축 해제 함수
# ======================================================
def download_and_extract(object_name):
    """YCB 오브젝트를 다운로드하고 압축 해제"""
    
    # URL 생성
    tar_filename = f"{object_name}_berkeley_meshes.tgz"
    url = f"{BASE_URL}/{object_name}/{tar_filename}"
    
    # 로컬 경로
    tar_path = os.path.join(OUTPUT_FOLDER, tar_filename)
    extract_path = os.path.join(OUTPUT_FOLDER, object_name)
    
    # 이미 압축 해제된 폴더가 있으면 스킵
    obj_file = os.path.join(extract_path, "poisson", "textured.obj")
    if os.path.exists(obj_file):
        print(f"[SKIP] {object_name} - 이미 존재합니다.")
        return
    
    print(f"[INFO] 다운로드 중: {object_name}")
    print(f"       URL: {url}")
    
    # 다운로드
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r       진행률: {percent:.1f}%", end='')
        
        print()  # 줄바꿈
        print(f"[INFO] 다운로드 완료: {tar_filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 다운로드 실패: {e}")
        return
    
    # 압축 해제
    print(f"[INFO] 압축 해제 중: {tar_filename}")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=OUTPUT_FOLDER)
        print(f"[DONE] 압축 해제 완료: {object_name}")
        
        # tar 파일 삭제
        os.remove(tar_path)
        print(f"[INFO] tar 파일 삭제: {tar_filename}")
        
    except Exception as e:
        print(f"[ERROR] 압축 해제 실패: {e}")
        return
    
    # OBJ 파일 확인
    if os.path.exists(obj_file):
        print(f"[OK] OBJ 파일 확인: {obj_file}")
    else:
        print(f"[WARN] OBJ 파일을 찾을 수 없습니다: {obj_file}")


# ======================================================
# 전체 오브젝트 다운로드
# ======================================================
print("==========================================")
print(f"총 {len(YCB_OBJECTS)}개의 YCB 오브젝트 다운로드 시작")
print("==========================================")

for obj_name in YCB_OBJECTS:
    print()
    download_and_extract(obj_name)

print()
print("==========================================")
print("✓ 모든 YCB 오브젝트 다운로드 완료")
print("==========================================")
