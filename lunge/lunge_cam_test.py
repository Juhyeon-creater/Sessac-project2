import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import joblib
from collections import deque

# --- 설정 ---
MODEL_PATH = 'lunge/lunge_lstm_model.h5'
SCALER_PATH = 'lunge/scaler.pkl'
SEQUENCE_LENGTH = 30  # LSTM 학습 시 설정한 길이

# 1. 모델 및 스케일러 로드
print("모델 로딩 중...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
yolo_model = YOLO('yolo11s-pose.pt')

# 2. 유틸리티 함수 (각도 계산)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

# 3. 웹캠 실행
cap = cv2.VideoCapture(0) # 0번은 기본 웹캠
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) # 30개만 유지하는 큐(Queue)

print("웹캠 시작! (종료하려면 'q'를 누르세요)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 거울 모드 (좌우 반전)
    frame = cv2.flip(frame, 1)
    
    # YOLO 추론
    results = yolo_model(frame, verbose=False)
    
    # 화면에 그리기 준비
    display_text = "Waiting for pose..."
    color = (0, 0, 255) # 기본 빨강

    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        # 가장 큰 사람 한 명만
        kpts = results[0].keypoints.xyn[0].cpu().numpy()
        
        # 좌표 추출 (인덱스 주의: 11:L_Hip, 12:R_Hip, 13:L_Knee, 14:R_Knee, 15:L_Ankle, 16:R_Ankle)
        # 0이 아닌지(감지됨) 확인
        if kpts[13][0] > 0 and kpts[14][0] > 0:
            l_hip, l_knee, l_ankle = kpts[11], kpts[13], kpts[15]
            r_hip, r_knee, r_ankle = kpts[12], kpts[14], kpts[16]

            # Feature 계산
            angle_l_knee = calculate_angle(l_hip, l_knee, l_ankle)
            angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)
            ankle_dist_x = abs(l_ankle[0] - r_ankle[0])
            l_hip_y = l_hip[1]
            r_hip_y = r_hip[1]

            # 현재 프레임 데이터 (1, 5)
            current_feature = np.array([[angle_l_knee, angle_r_knee, l_hip_y, r_hip_y, ankle_dist_x]])
            
            # 정규화 (Scaler 적용)
            current_feature = scaler.transform(current_feature)

            # 버퍼에 추가
            sequence_buffer.append(current_feature[0])

            # 시각화 (뼈대 그리기)
            for kp in [l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]:
                x, y = int(kp[0] * frame.shape[1]), int(kp[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # --- LSTM 예측 (데이터가 30개 찼을 때만) ---
    if len(sequence_buffer) == SEQUENCE_LENGTH:
        # 모델 입력 형태 변환: (1, 30, 5)
        input_data = np.expand_dims(list(sequence_buffer), axis=0)
        
        prediction = model.predict(input_data, verbose=0)[0][0]
        
        # 점수 변환 (0:Good ~ 1:Bad 이므로 반대로)
        score = (1 - prediction) * 100
        
        if score > 50:
            status = f"GOOD ({score:.1f})"
            color = (0, 255, 0) # 초록
        else:
            status = f"BAD ({score:.1f})"
            color = (0, 0, 255) # 빨강
            
        display_text = status

    # 텍스트 출력
    cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1) # 배경 박스
    cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Lunge AI Coach', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()