import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# --- 1. 모델 로드 (처음 한 번만 실행됨) ---
print("모델을 로딩 중입니다...")
model = YOLO('yolov8n-pose.pt') # 가장 가벼운 모델 사용

# --- 2. 분석 로직 함수들 (작성자님 로직 적용) ---
def calculate_metrics(row):
    """실시간 좌표에서 각도와 기울기를 계산"""
    # 1. 다리 각도 (Left Leg Angle)
    # y좌표는 아래로 갈수록 커지므로 -(ankle - hip)
    dy = -(row['Left_Ankle_y'] - row['Left_Hip_y']) 
    dx = row['Left_Ankle_x'] - row['Left_Hip_x']
    angle = np.degrees(np.arctan2(dy, dx))
    
    # 2. 골반 기울기 (절댓값)
    pelvis_slope = abs(row['Right_Hip_y'] - row['Left_Hip_y'])
    
    return angle, pelvis_slope

# --- 3. 메인 실행부 ---
def run_app():
    # 웹캠 켜기 (0번은 기본 카메라)
    cap = cv2.VideoCapture(0)
    
    # 녹화 상태 변수
    is_recording = False
    recorded_data = []
    
    print("="*50)
    print("🎥 시스템 시작!")
    print("👉 [R] 키: 녹화 시작 / 중지")
    print("👉 [Q] 키: 프로그램 종료 및 저장")
    print("="*50)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 1. YOLO 추론 (Inference)
        results = model(frame, verbose=False, conf=0.5) # 신뢰도 0.5 이상만
        
        # 화면에 뼈대 그리기 (YOLO 내장 기능)
        annotated_frame = results[0].plot()

        # 사람을 감지했을 경우에만 데이터 처리
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            # 첫 번째 사람(Index 0)의 데이터만 가져옴
            # (CPU로 옮기고 numpy 배열로 변환)
            keypoints = results[0].keypoints.data[0].cpu().numpy() 
            
            # --- [중요] 실시간 데이터 매핑 ---
            # YOLO 인덱스: 5(L_Shoulder), 6(R_Shoulder), 11(L_Hip), 12(R_Hip)...
            # 필요한 좌표만 뽑아서 변수로 만듦
            
            # 좌표 추출 (x, y, conf)
            l_hip = keypoints[11]
            r_hip = keypoints[12]
            l_ankle = keypoints[15]
            l_sh = keypoints[5]
            r_sh = keypoints[6]

            # 정규화 로직 (Scale Factor 계산)
            neck_x, neck_y = (l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2
            pelvis_x, pelvis_y = (l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2
            
            torso_len = np.sqrt((neck_x - pelvis_x)**2 + (neck_y - pelvis_y)**2)
            if torso_len == 0: torso_len = 1 # 에러 방지
            
            # 임시 딕셔너리 생성 (정규화 적용)
            current_data = {
                'Frame_Time': datetime.now().strftime('%H:%M:%S.%f'),
                'Left_Hip_x': (l_hip[0] - pelvis_x) / torso_len,
                'Left_Hip_y': (l_hip[1] - pelvis_y) / torso_len,
                'Right_Hip_y': (r_hip[1] - pelvis_y) / torso_len,
                'Left_Ankle_x': (l_ankle[0] - pelvis_x) / torso_len,
                'Left_Ankle_y': (l_ankle[1] - pelvis_y) / torso_len,
            }
            
            # 각도/기울기 계산
            angle, slope = calculate_metrics(current_data)
            
            # 화면에 실시간 수치 표시 (HUD)
            cv2.putText(annotated_frame, f"Angle: {angle:.1f}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Slope: {slope:.3f}", (50, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # --- [녹화 중일 때만 저장] ---
            if is_recording:
                # 저장할 데이터 추가 (분석 결과까지 포함)
                save_row = current_data.copy()
                save_row['Leg_Angle'] = angle
                save_row['Pelvis_Slope'] = slope
                recorded_data.append(save_row)
                
                # 녹화 중 표시 (빨간 동그라미)
                cv2.circle(annotated_frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "REC", (55, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 화면 출력
        cv2.imshow('Pilates AI Coach', annotated_frame)

        # 키보드 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Q 누르면 종료
            break
        elif key == ord('r'): # R 누르면 녹화 토글
            is_recording = not is_recording
            status = "시작" if is_recording else "중지"
            print(f"⏺️ 녹화 {status}!")

    # --- 종료 후 처리 ---
    cap.release()
    cv2.destroyAllWindows()
    
    if recorded_data:
        # CSV 저장
        df = pd.DataFrame(recorded_data)
        filename = f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ 저장 완료! 파일명: {filename}")
        print(f"총 {len(df)} 프레임이 저장되었습니다.")
        
        # 간단한 리포트 출력
        print("\n[Today's Report]")
        print(f"평균 다리 각도: {df['Leg_Angle'].mean():.1f}도")
        print(f"최대 골반 기울기: {df['Pelvis_Slope'].max():.3f}")
    else:
        print("\n⚠️ 녹화된 데이터가 없습니다.")

# 실행
if __name__ == '__main__':
    run_app()