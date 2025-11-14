import pandas as pd
import numpy as np
import os
import glob

def normalize_keypoints_named(df):
    """
    컬럼명이 'part_x', 'part_y' 형태일 때 몸통 길이 기준으로 정규화하는 함수
    """
    df_norm = df.copy()
    
    # 1. 기준점 계산을 위한 컬럼명 정의
    l_sh_x, l_sh_y = 'Left_Shoulder_x', 'Left_Shoulder_y'
    r_sh_x, r_sh_y = 'Right_Shoulder_x', 'Right_Shoulder_y'
    l_hip_x, l_hip_y = 'Left_Hip_x', 'Left_Hip_y'
    r_hip_x, r_hip_y = 'Right_Hip_x', 'Right_Hip_y'
    
    # 2. 중요 부위 중점(Center) 계산
    # 어깨 중점 (Neck)
    neck_x = (df[l_sh_x] + df[r_sh_x]) / 2
    neck_y = (df[l_sh_y] + df[r_sh_y]) / 2
    
    # 골반 중점 (Pelvis)
    pelvis_x = (df[l_hip_x] + df[r_hip_x]) / 2
    pelvis_y = (df[l_hip_y] + df[r_hip_y]) / 2
    
    # 3. 몸통 길이(Scale Factor) 계산
    torso_length = np.sqrt((neck_x - pelvis_x)**2 + (neck_y - pelvis_y)**2)
    torso_length = torso_length.replace(0, np.nan).fillna(1) # 0으로 나누기 방지

    # 4. 정규화 수행 (전체 부위 리스트)
    # YOLO 표준 17개 부위 이름 리스트
    body_parts = [
        'Nose', 'Left_Eye', 'Right_Eye', 'Left_Ear', 'Right_Ear',
        'Left_Shoulder', 'Right_Shoulder', 
        'Left_Elbow', 'Right_Elbow', 
        'Left_Wrist', 'Right_Wrist', 
        'Left_Hip', 'Right_Hip', 
        'Left_Knee', 'Right_Knee', 
        'Left_Ankle', 'Right_Ankle'
        ]
    
    for part in body_parts:
        col_x = f'{part}_x' # 예: Left_Knee_x
        col_y = f'{part}_y' # 예: Left_Knee_y
        
        # 해당 컬럼이 데이터에 존재할 경우에만 계산
        if col_x in df.columns and col_y in df.columns:
            # 결과 컬럼명 예: norm_left_knee_x
            df_norm[f'norm_{part}_x'] = (df[col_x] - pelvis_x) / torso_length
            df_norm[f'norm_{part}_y'] = (df[col_y] - pelvis_y) / torso_length
            
    return df_norm

#keypoint csv 파일이 존재하는 폴더
input_folder = 'keypoint_LegPull'

# output 폴더
output_folder = os.path.join(input_folder,'norm')
os.makedirs(output_folder, exist_ok=True)

all_files = glob.glob(os.path.join(input_folder,'*.csv'))
print(f"총 {len(all_files)}개의 파일을 찾았습니다. 처리를 시작합니다...")

success_count = 0
error_count = 0

for file_path in all_files:
    try:
        # 1. 파일 읽기
        df = pd.read_csv(file_path)
        
        # 2. 정규화 함수 적용
        df_normalized = normalize_keypoints_named(df)
        
        # 3. 파일명 추출 및 저장 경로 설정
        base_name = os.path.basename(file_path)
        save_path = os.path.join(output_folder, f"norm_{base_name}")
        
        # 4. 저장
        df_normalized.to_csv(save_path, index=False)
        success_count += 1
        
        # 진행 상황 출력 (10개 단위로만 출력하여 콘솔 도배 방지)
        if success_count % 10 == 0:
            print(f"{success_count}개 파일 처리 완료...")
            
    except Exception as e:
        print(f"❌ 오류 발생 파일: {base_name}")
        print(f"에러 내용: {e}")
        error_count += 1

print("-" * 30)
print(f"작업 완료!")
print(f"성공: {success_count}개")
print(f"실패: {error_count}개")
print(f"저장 위치: {output_folder}")