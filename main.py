import os
import sys
import subprocess
import urllib.request
import argparse
from pathlib import Path

# ======================================================
# 설정 변수
# ======================================================
# Blender 실행 파일 경로 (기본값: PATH에서 찾음)
# 예: "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
BLENDER_PATH = "blender"

# ======================================================
# USD 파일 목록 (이름: URL)
# ======================================================
USD_FILES = [
    {
        "name": "010_potted_meat_can.usd",
        "url": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd"
    },
    {
        "name": "011_banana.usd",
        "url": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/011_banana.usd"
    },
    {
        "name": "040_large_marker.usd",
        "url": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/040_large_marker.usd"
    },
    {
        "name": "005_tomato_soup_can.usd",
        "url": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd"
    }
]

# ======================================================
# 경로 설정
# ======================================================
SCRIPT_DIR = Path(__file__).parent
USD_DIR = SCRIPT_DIR / "assets" / "ycb_usd"
OBJ_DIR = SCRIPT_DIR / "assets" / "ycb_obj"

# ======================================================
# 1. USD 파일 다운로드
# ======================================================
def download_usd_files():
    """USD 파일 다운로드"""
    print("\n" + "="*60)
    print("STEP 1: USD 파일 다운로드")
    print("="*60)
    
    # 디렉토리 생성
    USD_DIR.mkdir(parents=True, exist_ok=True)
    
    for file_info in USD_FILES:
        file_name = file_info["name"]
        file_url = file_info["url"]
        file_path = USD_DIR / file_name
        
        if file_path.exists():
            print(f"[SKIP] {file_name} 이미 존재")
            continue
        
        print(f"[DOWNLOAD] {file_name}")
        try:
            urllib.request.urlretrieve(file_url, file_path)
            print(f"[OK] {file_name} 다운로드 완료")
        except Exception as e:
            print(f"[ERROR] {file_name} 다운로드 실패: {e}")
            return False
    
    print("\n✓ USD 파일 다운로드 완료\n")
    return True


# ======================================================
# 2. USD → OBJ 변환
# ======================================================
def convert_usd_to_obj(blender_path=BLENDER_PATH):
    """Blender를 사용하여 USD를 OBJ로 변환"""
    print("\n" + "="*60)
    print("STEP 2: USD → OBJ 변환")
    print("="*60)
    
    usd_to_obj_script = SCRIPT_DIR / "usd_to_obj.py"
    
    if not usd_to_obj_script.exists():
        print(f"[ERROR] {usd_to_obj_script} 파일을 찾을 수 없습니다.")
        return False
    
    # Blender 실행
    cmd = [blender_path, "--background", "--python", str(usd_to_obj_script)]
    
    print(f"[RUN] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✓ USD → OBJ 변환 완료\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Blender 실행 실패: {e}")
        return False
    except FileNotFoundError:
        print(f"[ERROR] Blender를 찾을 수 없습니다: {blender_path}")
        print("--blender-path 옵션으로 Blender 경로를 지정하거나 PATH에 추가하세요.")
        return False


# ======================================================
# 3. BlenderProc 데이터셋 생성
# ======================================================
def generate_dataset(num_scenes=10):
    """BlenderProc로 데이터셋 생성"""
    print("\n" + "="*60)
    print("STEP 3: BlenderProc 데이터셋 생성")
    print("="*60)
    
    generate_script = SCRIPT_DIR / "generate_dataset.py"
    
    if not generate_script.exists():
        print(f"[ERROR] {generate_script} 파일을 찾을 수 없습니다.")
        return False
    
    cmd = ["blenderproc", "run", str(generate_script), "--num_scenes", str(num_scenes)]
    
    print(f"[RUN] {' '.join(cmd)}")
    
    # Blender는 정상 종료시에도 -1을 반환할 수 있으므로 exit code 무시
    subprocess.run(cmd, capture_output=False)
    
    # 출력 파일이 생성되었는지 확인
    output_dir = SCRIPT_DIR / "dataset" / "raw"
    if output_dir.exists():
        scene_dirs = list(output_dir.glob("scene_*"))
        if len(scene_dirs) >= num_scenes:
            print("\n✓ 데이터셋 생성 완료\n")
            return True
    
    print("\n[ERROR] 출력 파일이 생성되지 않았습니다.")
    return False


# ======================================================
# 4. HDF5 → YOLO 포맷 변환
# ======================================================
def convert_to_yolo():
    """HDF5를 YOLO 포맷으로 변환"""
    print("\n" + "="*60)
    print("STEP 4: HDF5 → YOLO 포맷 변환")
    print("="*60)
    
    convert_script = SCRIPT_DIR / "convert_to_yolo.py"
    
    if not convert_script.exists():
        print(f"[ERROR] {convert_script} 파일을 찾을 수 없습니다.")
        return False
    
    cmd = [sys.executable, str(convert_script)]
    
    print(f"[RUN] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✓ YOLO 포맷 변환 완료\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 변환 스크립트 실행 실패: {e}")
        return False


# ======================================================
# 5. YOLO 모델 학습
# ======================================================
def train_yolo():
    """YOLO 모델 학습"""
    print("\n" + "="*60)
    print("STEP 5: YOLO 모델 학습")
    print("="*60)
    
    train_script = SCRIPT_DIR / "train_yolo.py"
    
    if not train_script.exists():
        print(f"[ERROR] {train_script} 파일을 찾을 수 없습니다.")
        return False
    
    cmd = [sys.executable, str(train_script)]
    
    print(f"[RUN] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✓ YOLO 학습 완료\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 학습 스크립트 실행 실패: {e}")
        return False


# ======================================================
# 메인 실행
# ======================================================
def main():
    """전체 워크플로우 실행"""
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(
        description='BlenderProc → YOLO 전체 워크플로우 자동화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python main.py
  python main.py --num-scenes 20
  python main.py --blender-path "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
  python main.py --skip-download --skip-convert
        """
    )
    
    parser.add_argument(
        '--blender-path',
        type=str,
        default=BLENDER_PATH,
        help=f'Blender 실행 파일 경로 (기본값: {BLENDER_PATH})'
    )
    parser.add_argument(
        '--num-scenes',
        type=int,
        default=10,
        help='생성할 씬 개수 (기본값: 10)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='USD 파일 다운로드 단계 건너뛰기'
    )
    parser.add_argument(
        '--skip-convert',
        action='store_true',
        help='USD → OBJ 변환 단계 건너뛰기'
    )
    parser.add_argument(
        '--skip-generate',
        action='store_true',
        help='데이터셋 생성 단계 건너뛰기'
    )
    parser.add_argument(
        '--skip-yolo-convert',
        action='store_true',
        help='YOLO 변환 단계 건너뛰기'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='학습 단계 건너뛰기'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BlenderProc → YOLO 전체 워크플로우 자동화")
    print("="*60)
    print(f"Blender 경로: {args.blender_path}")
    print(f"씬 개수: {args.num_scenes}")
    print("="*60)
    
    # 단계별 실행
    steps = [
        ("USD 파일 다운로드", download_usd_files, args.skip_download),
        ("USD → OBJ 변환", lambda: convert_usd_to_obj(args.blender_path), args.skip_convert),
        ("BlenderProc 데이터셋 생성", lambda: generate_dataset(num_scenes=args.num_scenes), args.skip_generate),
        ("HDF5 → YOLO 변환", convert_to_yolo, args.skip_yolo_convert),
        ("YOLO 모델 학습", train_yolo, args.skip_train),
    ]
    
    for i, (step_name, step_func, skip) in enumerate(steps, 1):
        if skip:
            print(f"\n{'='*60}")
            print(f"[{i}/{len(steps)}] {step_name} - SKIPPED")
            print(f"{'='*60}")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(steps)}] {step_name}")
        print(f"{'='*60}")
        
        success = step_func()
        
        if not success:
            print(f"\n[FAILED] {step_name} 단계에서 오류가 발생했습니다.")
            print("워크플로우를 중단합니다.")
            sys.exit(1)
    
    # 완료
    print("\n" + "="*60)
    print("🎉 전체 워크플로우 완료!")
    print("="*60)
    print("\n학습된 모델 위치:")
    print("  - runs/detect/train/weights/best.pt")
    print("  - runs/detect/train/weights/last.pt")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
