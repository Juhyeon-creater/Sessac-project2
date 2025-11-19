import cv2
import numpy as np
from ultralytics import YOLO
import requests
import math


# ===============================
# 요가 각도 계산
# ===============================
TARGET_ANGLE_MIN = 7.77
TARGET_ANGLE_MAX = 14.75

def calculate_leg_angle(kps):
    l_conf = kps[15][2]
    r_conf = kps[16][2]

    if l_conf >= r_conf:
        hip = kps[11][:2]
        ankle = kps[15][:2]
    else:
        hip = kps[12][:2]
        ankle = kps[16][:2]

    dy = -(ankle[1] - hip[1])
    dx = abs(ankle[0] - hip[0])
    angle = np.degrees(np.arctan2(dy, dx))

    return angle, hip, ankle


# ===============================
# MPU6050 센서
# ===============================
SENSOR_URL = "http://192.168.4.1"

def get_sensor_data():
    try:
        r = requests.get(SENSOR_URL, timeout=0.05)
        return r.json()
    except:
        return None

def compute_pelvis_from_mpu(sensor_data, threshold=3.0):
    if sensor_data is None:
        return None
    m1 = sensor_data.get("mpu1")
    m2 = sensor_data.get("mpu2")

    if not (m1 and m2):
        return None

    def roll_from_mpu(m):
        ax = m["AcX"] / 16384
        ay = m["AcY"] / 16384
        az = m["AcZ"] / 16384
        return math.degrees(math.atan2(ay, az))

    r1 = roll_from_mpu(m1)
    r2 = roll_from_mpu(m2)
    diff = r1 - r2

    if abs(diff) < threshold:
        status = "Pelvis: LEVEL"
    elif diff > 0:
        status = "Pelvis: RIGHT DOWN"
    else:
        status = "Pelvis: LEFT DOWN"

    return {"r1": r1, "r2": r2, "diff": diff, "status": status}


# ===============================
# 메인 실행
# ===============================
def run_hundred_coach():

    print("⏳ YOLO 모델 로딩 중...")
    model = YOLO("yolo11s-pose.pt")

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    print("✅ 시작!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) YOLO 추론
        results = model(frame, verbose=False, conf=0.5)

        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kps = results[0].keypoints.data[0].cpu().numpy()

            # 다리 각도 계산
            angle, hip_xy, ankle_xy = calculate_leg_angle(kps)
            good = TARGET_ANGLE_MIN <= angle <= TARGET_ANGLE_MAX

            color = (0,255,0) if good else (0,0,255)
            status = f"{'GOOD' if good else 'BAD'} ({angle:.1f})"

            cv2.line(frame, tuple(hip_xy.astype(int)), tuple(ankle_xy.astype(int)), color, 4)

            cv2.putText(frame, status, (int(hip_xy[0]), int(hip_xy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 2) MPU6050 센서도 화면에 표시
        sensor_data = get_sensor_data()
        pelvis = compute_pelvis_from_mpu(sensor_data)

        if pelvis:
            text = f"{pelvis['status']} (R:{pelvis['r1']:.1f} L:{pelvis['r2']:.1f} Δ:{pelvis['diff']:.1f})"
            h, w, _ = frame.shape
            cv2.putText(frame, text, (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # 3) 화면 출력
        cv2.imshow("Hundred AI Coach", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hundred_coach()
