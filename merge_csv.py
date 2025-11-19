import pandas as pd
import numpy as np
import os
import glob

# --- 1. 정규화 함수 (기존과 동일) ---
def normalize_keypoints_named(df):
    df_norm = df.copy()
    l_sh_x, l_sh_y = 'Left_Shoulder_x', 'Left_Shoulder_y'
    r_sh_x, r_sh_y = 'Right_Shoulder_x', 'Right_Shoulder_y'
    l_hip_x, l_hip_y = 'Left_Hip_x', 'Left_Hip_y'
    r_hip_x, r_hip_y = 'Right_Hip_x', 'Right_Hip_y'
    
    # 예외처리: 필수 컬럼이 없으면 그냥 원본 리턴
    if l_sh_x not in df.columns: return df

    neck_x = (df[l_sh_x] + df[r_sh_x]) / 2
    neck_y = (df[l_sh_y] + df[r_sh_y]) / 2
    pelvis_x = (df[l_hip_x] + df[r_hip_x]) / 2
    pelvis_y = (df[l_hip_y] + df[r_hip_y]) / 2
    
    torso_length = np.sqrt((neck_x - pelvis_x)**2 + (neck_y - pelvis_y)**2)
    torso_length = torso_length.replace(0, np.nan).fillna(1) 

    body_parts = [
        'Nose', 'Left_Eye', 'Right_Eye', 'Left_Ear', 'Right_Ear',
        'Left_Shoulder', 'Right_Shoulder', 
        'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist', 
        'Left_Hip', 'Right_Hip', 'Left_Knee', 'Right_Knee', 
        'Left_Ankle', 'Right_Ankle'
    ]
    
    for part in body_parts:
        col_x, col_y = f'{part}_x', f'{part}_y'
        if col_x in df.columns and col_y in df.columns:
            df_norm[f'norm_{part}_x'] = (df[col_x] - pelvis_x) / torso_length
            df_norm[f'norm_{part}_y'] = (df[col_y] - pelvis_y) / torso_length
            
    return df_norm

# --- 2. 메인 실행부 (정보 추출 및 병합) ---

input_folder = 'keypoint_LegPull'
all_files = glob.glob(os.path.join(input_folder, "*.csv"))

merged_list = [] # 모든 데이터프레임을 담을 리스트

print(f"총 {len(all_files)}개 파일 병합 시작...")

for file_path in all_files:
    try:
        # 1. 파일 읽기
        df = pd.read_csv(file_path)
        
        # 2. 파일명에서 정보 추출 (Parsing)
        filename = os.path.basename(file_path)
        clean_name = filename.replace('.csv', '')
        
        # 언더바(_) 기준으로 자르기
        # 예: 필라테스_가산_A__Leg Pull_고급_actorP061_...
        parts = clean_name.split('_')
        
        # [중요] 파일명 규칙에 따라 인덱스 번호가 다를 수 있음. 
        # 아래 인덱스는 보내주신 예시 기준입니다.
        # parts[0]: 필라테스, parts[1]: 가산, parts[4]: Leg Pull...
        
        place = parts[1]   # 가산
        level = parts[4]   # 고급 (만약 'Leg Pull' 뒤에 바로 오면 인덱스 확인 필요)
        
        # Person_ID 찾기 ('actor'로 시작하는 부분 찾기 - 안전한 방법)
        person_id = "Unknown"
        for part in parts:
            if part.startswith('actor'):
                person_id = part
                break
        
        # 3. 추출한 정보를 컬럼으로 추가 (Categorical Data)
        df['Place'] = place
        df['Level'] = level
        df['Person_ID'] = person_id
        df['Source_File'] = filename # 나중에 원본 확인할 때 유용
        
        # 4. 정규화 수행
        df_norm = normalize_keypoints_named(df)
        
        # 5. 리스트에 추가
        merged_list.append(df_norm)
        
    except Exception as e:
        print(f"Skipping {filename}: {e}")

# 6. 하나로 합치기 (Concat)
if merged_list:
    final_df = pd.concat(merged_list, axis=0, ignore_index=True)
    
    # 7. 최종 파일 저장
    save_path = os.path.join(input_folder, "Grand_Merged_Dataset.csv")
    final_df.to_csv(save_path, index=False)
    
    print("-" * 30)
    print("✅ 병합 완료!")
    print(f"전체 데이터 크기: {final_df.shape}") # (행 개수, 열 개수)
    print(f"저장된 파일: {save_path}")
    
    # 데이터 확인용
    print("\n[데이터 미리보기]")
    print(final_df[['Person_ID', 'Level', 'Place']].head())
else:
    print("병합할 데이터가 없습니다.")