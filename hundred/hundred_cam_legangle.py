import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. 설정값 (Thresholds) ---
TARGET_ANGLE_MIN = 7.77   # 하한값
TARGET_ANGLE_MAX = 14.75  # 상한값

# --- 2. 분석 함수 ---
def calculate_leg_angle(kps):
    """
    YOLO Keypoints(17, 3)를 받아 다리 각도를 계산
    - 스마트 다리 선택 (더 잘 보이는 다리 사용)
    - 수평선 기준 각도 계산
    """
    # 관절 인덱스 (YOLOv8 Pose 기준)
    # 11:Left_Hip, 12:Right_Hip, 15:Left_Ankle, 16:Right_Ankle
    
    # 신뢰도(conf) 추출
    l_conf = kps[15][2]
    r_conf = kps[16][2]
    
    # (1) 스마트 다리 선택
    # 왼쪽 발목 신뢰도가 더 높거나 같으면 왼쪽 사용, 아니면 오른쪽 사용
    if l_conf >= r_conf:
        hip = kps[11][:2]   # [x, y]
        ankle = kps[15][:2] # [x, y]
        side = "Left"
    else:
        hip = kps[12][:2]
        ankle = kps[16][:2]
        side = "Right"

    # (2) 각도 계산 (수평선 기준)
    # Y축은 아래로 갈수록 커지므로, 위로 들면 y가 작아짐 -> dy를 반전(-)
    dy = -(ankle[1] - hip[1]) 
    dx = np.abs(ankle[0] - hip[0]) # 방향 상관없이 수평 거리만 (절댓값)
    
    angle = np.degrees(np.arctan2(dy, dx))
    
    return angle, side, hip, ankle

# --- 3. 메인 실행부 ---
def run_hundred_coach():
    # 모델 로드 (가벼운 모델 사용 추천)
    print("⏳ 모델 로딩 중...")
    model = YOLO('yolo11s-pose.pt') 
    
    # 웹캠 연결 (0번: 기본 카메라)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    print("✅ 시스템 시작! 카메라 앞에서 Hundred 자세를 취하세요.")
    print(f"🎯 목표 각도: {TARGET_ANGLE_MIN}° ~ {TARGET_ANGLE_MAX}°")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. YOLO 추론
        results = model(frame, verbose=False, conf=0.5)
        
        # 2. 기본적으로 스켈레톤을 그리지 않고 원본 프레임 사용 (우리가 직접 그리기 위해)
        # 만약 YOLO 기본 그림 위에 덧칠하려면: frame = results[0].plot()
        
        # 사람이 감지되었는지 확인
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            # 첫 번째 사람만 분석
            kps = results[0].keypoints.data[0].cpu().numpy() # (17, 3)
            
            # 각도 계산
            angle, side, hip_xy, ankle_xy = calculate_leg_angle(kps)
            
            # --- [판단 로직] ---
            is_good = (angle >= TARGET_ANGLE_MIN) and (angle <= TARGET_ANGLE_MAX)
            
            # --- [시각화 로직] ---
            if is_good:
                color = (0, 255, 0) # 초록색 (BGR)
                status_text = f"GOOD ({angle:.1f})"
                msg = ""
            else:
                color = (0, 0, 255) # 빨간색 (BGR)
                status_text = f"BAD ({angle:.1f})"
                
                # 피드백 메시지 구체화
                if angle < TARGET_ANGLE_MIN:
                    msg = "UP! Leg is too low"
                else:
                    msg = "DOWN! Leg is too high"

            # 1. 스켈레톤(다리 라인) 직접 그리기 (색상 변경 적용)
            # 힙 -> 발목 선 그리기
            h_pt = (int(hip_xy[0]), int(hip_xy[1]))
            a_pt = (int(ankle_xy[0]), int(ankle_xy[1]))
            
            cv2.line(frame, h_pt, a_pt, color, 4) # 선 두께 4
            cv2.circle(frame, h_pt, 6, color, -1)
            cv2.circle(frame, a_pt, 6, color, -1)
            
            # 2. 화면에 정보 표시 (HUD)
            # 각도 표시
            cv2.putText(frame, status_text, (h_pt[0] + 10, h_pt[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 경고 메시지 (화면 중앙 상단)
            if not is_good:
                cv2.putText(frame, "WARNING!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, msg, (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 화면 테두리에 빨간색 박스 쳐서 경고 강조
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)

        # 화면 출력
        cv2.imshow('Hundred AI Coach', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hundred_coach()