from datasets import load_dataset

# 1. 데이터셋 로드 (이미 하신 부분)
ds = load_dataset("guyuchao/UCF101")

# 2. 'Lunges'가 몇 번 라벨인지 확인하는 함수
# (features 안에 라벨 이름들이 들어있습니다)
labels = ds['train'].features['label'].names
target_label = "Lunges"

if target_label in labels:
    label_id = labels.index(target_label)
    print(f"✅ '{target_label}'의 라벨 ID는 [{label_id}]번 입니다.")
else:
    print(f"❌ '{target_label}' 라벨을 찾을 수 없습니다.")
    label_id = -1

# 3. 라벨 ID를 이용해서 데이터 필터링 (Lunges만 남기기)
if label_id != -1:
    # train 데이터에서 필터링
    lunges_dataset = ds['train'].filter(lambda x: x['label'] == label_id)
    
    print(f"\n--- 결과 확인 ---")
    print(f"전체 비디오 개수: {len(ds['train'])}")
    print(f"추출된 런지 비디오 개수: {len(lunges_dataset)}")
    
    # 4. 실제 데이터 예시 확인 (경로와 라벨)
    print("\n[샘플 데이터 3개 확인]")
    for i in range(3):
        print(lunges_dataset[i])

else:
    # 만약 라벨 이름 매칭이 안되면, 파일 이름(video_path)으로 무식하게 찾기
    print("\n⚠️ 라벨 ID 매칭 실패, 파일명으로 검색합니다.")
    lunges_dataset = ds['train'].filter(lambda x: "Lunges" in x['video_path'])
    print(f"파일명으로 찾은 개수: {len(lunges_dataset)}")