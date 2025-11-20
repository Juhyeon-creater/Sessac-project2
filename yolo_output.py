from ultralytics import YOLO
import cv2
import pandas as pd
import os
import time

input_folder = '.'                  #영상이 들어있는 폴더주소 (같은 폴더라면 .)
output_folder = 'keypoint_LegPull'  #결과넣을파일 (생성)


os.makedirs(output_folder, exist_ok= True)
video_exts = ('.mp4', '.avi', '.mov', '.mkv')

# 현재 폴더(.)에서 영상 파일만 골라내기
files = [f for f in os.listdir(input_folder) if f.lower().endswith(video_exts)]
total_files = len(files)



#  모델 로드

model = YOLO('yolo11x-pose.pt')



# 컬럼 이름 미리 정의 (COCO Keypoints 17개)
keypoint_names = [
    "Nose", "Left_Eye", "Right_Eye", "Left_Ear", "Right_Ear",
    "Left_Shoulder", "Right_Shoulder", "Left_Elbow", "Right_Elbow",
    "Left_Wrist", "Right_Wrist", "Left_Hip", "Right_Hip",
    "Left_Knee", "Right_Knee", "Left_Ankle", "Right_Ankle"
]
for i, filename in enumerate(files):
    # 영상경로 (다른 파일에 있을 시 경로로 수정해야함)
    video_path = os.path.join(input_folder, filename)
    
    csv_filename = os.path.splitext(filename)[0] + '.csv'
    save_path = os.path.join(output_folder, csv_filename)
    
    print(f"\n[{i+1}/{total_files}] 분석 중... {filename}")
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #window 설정
    show_video = True
    window_name = 'YOLO'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    video_data = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, verbose=False)
        
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            person_kpts = results[0].keypoints.data[0].cpu().numpy()
            
            frame_data = {
                "Frame": frame_idx
            }
            
            for k, kp in enumerate(person_kpts):
                x, y, conf = kp
                name = keypoint_names[k]
                frame_data[f"{name}_x"] = round(float(x), 2)
                frame_data[f"{name}_y"] = round(float(y), 2)
                frame_data[f"{name}_conf"] = round(float(conf), 2)
            
            video_data.append(frame_data)
        
        if show_video:
            annotated_frame = results[0].plot()
            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
        if frame_idx % 100 == 0: print(".", end="", flush=True)

    cap.release()
    if show_video: cv2.destroyAllWindows()
    
    if video_data:
        df = pd.DataFrame(video_data)
        df.to_csv(save_path, index=False)
        elapsed = time.time() - start_time
        print(f"\n   ✅ 저장 완료! ({elapsed:.1f}초) -> {output_folder}/{csv_filename}")
    else:
        print(f"\n   ⚠️ 데이터 없음 -> {filename}")

print("\n🎉 모든 작업 끝!")

# 5. CSV 파일로 저장 (Pandas 활용)
df = pd.DataFrame(all_data)
df.to_csv(output_csv, index=False)

print(f"'{output_csv}' 파일에 저장되었습니다.")