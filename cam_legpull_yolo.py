import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# --- 1. 모델 로드 ---
print("⏳ 모델을 불러오는 중입니다... (잠시만 기다려주세요)")
model = YOLO('yolo11n-pose.pt')

# --- 2. Leg Pull 전용 분석 함수 (이전 대화 내용 적용) ---
def analyze_leg_pull(keypoints):
    """
    YOLO Keypoints를 받아 Leg Pull 동작의 핵심 지표를 계산
    """
    # 관절 좌표 추출 (Tensor -> Numpy)
    # 5:L_Sh, 6:R_Sh, 11:L_Hip, 12:R_Hip, 15:L_Ankle, 16:R_Ankle
    kps = keypoints.cpu().numpy()
    
    l_sh, r_sh = kps[5][:2], kps[6][:2]
    l_hip, r_hip = kps[11][:2], kps[12][:2]
    l_ankle, r_ankle = kps[15][:2], kps[16][:2]

    # 정규화 스케일 (몸통 길이)
    neck = (l_sh + r_sh) / 2
    pelvis = (l_hip + r_hip) / 2
    torso_len = np.linalg.norm(neck - pelvis)
    if torso_len == 0: torso_len = 1.0

    # (1) 활성 다리 감지 (Y좌표가 더 작은 쪽이 들린 쪽)
    if l_ankle[1] < r_ankle[1]: 
        active_leg = "Left"
        # 각도 계산 (수평선 기준)
        dy = -(l_ankle[1] - l_hip[1])
        dx = l_ankle[0] - l_hip[0]
    else:
        active_leg = "Right"
        dy = -(r_ankle[1] - r_hip[1])
        dx = r_ankle[0] - r_hip[0]
    
    leg_angle = np.degrees(np.arctan2(dy, abs(dx)))

    # (2) 골반 안정성 (높이 차이 / 몸통길이)
    pelvis_diff = abs(l_hip[1] - r_hip[1]) / torso_len

    # (3) 몸통 처짐 (Body Line) - 지지하는 쪽 어깨-골반-발목 각도
    if active_leg == "Left":
        # 오른쪽이 지지하는 발
        vec_body = r_sh - r_hip
        vec_leg = r_ankle - r_hip
    else:
        # 왼쪽이 지지하는 발
        vec_body = l_sh - l_hip
        vec_leg = l_ankle - l_hip
        
    cosine = np.dot(vec_body, vec_leg) / (np.linalg.norm(vec_body) * np.linalg.norm(vec_leg))
    body_line = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return {
        "Active_Leg": active_leg,
        "Leg_Angle": leg_angle,
        "Pelvis_Stability": pelvis_diff,
        "Body_Line": body_line,
        "Torso_Len": torso_len, # 나중에 정규화 좌표 복원용
        "Pelvis_Center": pelvis # 나중에 정규화 좌표 복원용
    }

# --- 3. 메인 실행 루프 ---
def run_system():
    cap = cv2.VideoCapture(0) # 웹캠 연결
    is_recording = False
    record_list = []

    print("✅ 시스템 시작! 카메라 앞 2~3m 거리에서 전신이 나오게 서주세요.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # YOLO 추론
        results = model(frame, verbose=False, conf=0.5)
        annotated_frame = results[0].plot() # 뼈대 그리기

        # 사람이 감지되었을 때만 분석 수행
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            # 첫 번째 사람 데이터 가져오기
            kps_tensor = results[0].keypoints.data[0]
            
            # --- 분석 함수 호출 ---
            metrics = analyze_leg_pull(kps_tensor)
            
            # --- 화면에 실시간 피드백 표시 (UI) ---
            # 1. 골반 상태 (임계값 0.05)
            p_color = (0, 255, 0) # 초록색 (Good)
            p_msg = "Good"
            if metrics['Pelvis_Stability'] > 0.05:
                p_color = (0, 0, 255) # 빨간색 (Bad)
                p_msg = "Fix Pelvis!" # 골반 교정 필요
            
            # 2. 몸통 라인 (임계값 160도 미만이면 처짐)
            b_color = (0, 255, 0)
            if metrics['Body_Line'] < 160:
                b_color = (0, 0, 255)
                p_msg = "Core Fail!" # 코어 무너짐

            # 텍스트 출력
            cv2.putText(annotated_frame, f"Leg Angle: {metrics['Leg_Angle']:.1f}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.putText(annotated_frame, f"Pelvis: {metrics['Pelvis_Stability']:.3f} ({p_msg})", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, p_color, 2)
            
            cv2.putText(annotated_frame, f"Body Line: {metrics['Body_Line']:.1f}", (30, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, b_color, 2)

            # --- 녹화 (데이터 수집) ---
            if is_recording:
                # 분석된 지표 + 정규화된 좌표 저장
                data_row = metrics.copy()
                data_row['Time'] = datetime.now().strftime('%H:%M:%S.%f')
                # (필요하다면 여기에 kps 좌표들도 정규화해서 추가 가능)
                del data_row['Torso_Len'] # 저장할 땐 불필요한 정보 제거
                del data_row['Pelvis_Center']
                
                record_list.append(data_row)
                
                # 녹화 표시
                cv2.circle(annotated_frame, (width:=int(cap.get(3))-50, 50), 15, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "REC", (width-25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # 화면 출력
        cv2.imshow('Leg Pull AI Coach', annotated_frame)

        # 키 조작
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = not is_recording
            print(f"⏺️ 녹화 상태: {is_recording}")

    # 종료 및 저장
    cap.release()
    cv2.destroyAllWindows()

    if record_list:
        df = pd.DataFrame(record_list)
        filename = f"LegPull_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ [저장 완료] {filename}")
        print("이 파일을 태블로에 넣어서 분석하시면 됩니다!")
    else:
        print("\n⚠️ 저장된 데이터가 없습니다.")

if __name__ == "__main__":
    run_system()