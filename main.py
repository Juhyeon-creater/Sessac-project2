<<<<<<< HEAD
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import math
import time
import pandas as pd
from datetime import datetime # 파일명 및 시간 기록용

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
# MPU6050 센서 클라이언트
# ===============================
SENSOR_URL = "http://192.168.4.1"

def get_sensor_data():
    try:
        r = requests.get(SENSOR_URL, timeout=0.05)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.exceptions.RequestException:
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

    # [수정] Log Raw Data for deeper EDA
    return {"r1": r1, "r2": r2, "diff": diff, "status": status, "m1": m1, "m2": m2}


# ===============================
# 메인 실행 (녹화 기능 포함)
# ===============================
def run_hundred_coach():
    WINDOW_NAME = "Hundred AI Coach"
    
    # [추가] 로깅 및 녹화 변수
    recorded_data_log = []
    is_recording = False
    video_writer = None 

    # 1. 윈도우 생성 및 크기 설정
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720) 

    print("⏳ YOLO 모델 로딩 중...")
    model = YOLO("yolo11s-pose.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    print("✅ 시작!")

    # [추가] 프레임 정보 가져오기 (비디오 저장용)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        log_entry = {'Timestamp': time.time()}
        
        # 1) YOLO 추론
        results = model(frame, verbose=False, conf=0.5)
        
        # 뼈대 그림을 먼저 프레임에 입힙니다.
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            frame = results[0].plot(img=frame)
            
            kps = results[0].keypoints.data[0].cpu().numpy()

            # 다리 각도 계산
            angle, hip_xy, ankle_xy = calculate_leg_angle(kps)
            good = TARGET_ANGLE_MIN <= angle <= TARGET_ANGLE_MAX

            color = (0,255,0) if good else (0,0,255)
            status = f"{'GOOD' if good else 'BAD'} ({angle:.1f}°)"

            # [LOG] YOLO 데이터 기록
            log_entry['YOLO_Angle_deg'] = angle
            log_entry['YOLO_Status'] = status
            
            # (A) YOLO 분석 결과 표시
            cv2.putText(frame, f"YOLO Angle: {status}", (20, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
        # 2) MPU6050 센서도 화면에 표시
        sensor_data = get_sensor_data()
        pelvis = compute_pelvis_from_mpu(sensor_data)

        if pelvis:
            text = f"{pelvis['status']} (R:{pelvis['r1']:.1f} L:{pelvis['r2']:.1f} Δ:{pelvis['diff']:.1f})"
            h, w, _ = frame.shape
            
            # [LOG] MPU 데이터 기록
            log_entry['MPU_Pelvis_Status'] = pelvis['status']
            log_entry['MPU_Roll_Diff'] = pelvis['diff']
            log_entry['MPU1_Roll'] = pelvis['r1']
            log_entry['MPU2_Roll'] = pelvis['r2']
            
            # (Raw Accel/Gyro data for deeper analysis)
            if 'm1' in pelvis:
                log_entry['M1_AcX'] = pelvis['m1'].get('AcX', np.nan)
                log_entry['M2_GyY'] = pelvis['m2'].get('GyY', np.nan) 
            
            # (B) MPU 센서 데이터 표시
            cv2.putText(frame, text, (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2) # Yellow

        # --- [녹화 및 로깅 처리] ---
        h, w, _ = frame.shape
        if is_recording:
            # 3A. 비디오 프레임 저장
            if video_writer is not None:
                 video_writer.write(frame)
                 
            # 3B. 데이터 로그 저장
            if len(log_entry) > 1:
                recorded_data_log.append(log_entry)
                
            # 3C. REC 표시
            cv2.putText(frame, "REC", (w - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3) # Red REC


        # 4. 화면 출력 및 키 처리
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): # Q: 종료
            break
        elif key == ord('r'): # R: 녹화 시작/중지 토글
            is_recording = not is_recording
            
            if is_recording:
                # 녹화 시작 시 VideoWriter 초기화
                video_filename = f"Video_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
                
                print(f"--- 🎥 비디오 녹화 시작: {video_filename} ---")
            else:
                # 녹화 중지 시 VideoWriter 해제
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print(f"--- ⏸️ 비디오 녹화 중지. 파일이 저장됨. ---")
        # --- [녹화 처리 끝] ---


    # --- 최종 종료 처리 ---
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    # 데이터 CSV 저장
    if recorded_data_log:
        df = pd.DataFrame(recorded_data_log)
        filename = f"Data_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ 데이터 CSV 저장 완료! 파일명: {filename} ({len(df)} 프레임)")
    else:
        print("\n⚠️ 녹화된 데이터가 없습니다.")


if __name__ == "__main__":
    run_hundred_coach()   
=======
from machine import I2C, Pin
import network
import socket
import json
import time

# ==========================================================
#  MPU6050 x 2 + Raspberry Pi Pico W SoftAP 통합 예제
#  - Pico W가 SoftAP(와이파이 공유기) 역할 수행
#  - 클라이언트가 접속 요청을 보내면 두 개의 MPU6050 센서 데이터 반환
# ==========================================================

# ===============================
# ① I2C 설정 및 센서 스캔
# ===============================
i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=400000)   # I2C0, GP1=SCL, GP0=SDA, 400kHz

print("Scanning I2C bus...")
scan = i2c.scan()
print("Scan result:", [hex(a) for a in scan])

MPU_ADDR1 = 0x68   # AD0 = GND
MPU_ADDR2 = 0x69   # AD0 = 3.3V

if MPU_ADDR1 in scan:
    print("Found MPU #1 (0x68)")
else:
    print("MPU #1 (0x68) not detected")

if MPU_ADDR2 in scan:
    print("Found MPU #2 (0x69)")
else:
    print("MPU #2 (0x69) not detected")

# 실제로 초기화에 성공한 센서 주소들을 저장
mpu_addrs = []

for addr in (MPU_ADDR1, MPU_ADDR2):
    if addr in scan:
        try:
            # PWR_MGMT_1(0x6B) = 0 → sleep 해제
            i2c.writeto_mem(addr, 0x6B, bytes([0]))
            mpu_addrs.append(addr)
            print("MPU init OK at", hex(addr))
        except OSError:
            print("MPU init FAILED at", hex(addr))

print("Active MPU addrs:", [hex(a) for a in mpu_addrs])

# ===============================
# ② 센서 데이터 읽기 함수 (공통)
# ===============================
def safe_read(dev_addr, reg_addr):
    """
    지정한 MPU6050(dev_addr)의 레지스터(reg_addr)에서 2바이트를 안전하게 읽는다.
    - 통신 오류 발생 시 0 반환
    """
    try:
        data = i2c.readfrom_mem(dev_addr, reg_addr, 2)
        value = int.from_bytes(data, 'big')
        if value > 32768:   # 16비트 signed 변환
            value -= 65536
        return value
    except OSError:
        return 0

def read_one_mpu(dev_addr):
    """
    하나의 MPU6050(dev_addr)에 대해 가속도, 자이로, 온도 값을 dict로 반환
    """
    return {
        'AcX': safe_read(dev_addr, 0x3B),
        'AcY': safe_read(dev_addr, 0x3D),
        'AcZ': safe_read(dev_addr, 0x3F),
        'GyX': safe_read(dev_addr, 0x43),
        'GyY': safe_read(dev_addr, 0x45),
        'GyZ': safe_read(dev_addr, 0x47),
        'Temp': round(safe_read(dev_addr, 0x41) / 340 + 36.53, 2),
        'addr': hex(dev_addr),
    }

def read_all_sensors():
    """
    활성화된 모든 MPU6050(mpu_addrs 기준)을 읽어서
    mpu1, mpu2 형태의 dict로 반환
    """
    result = {}
    for idx, addr in enumerate(mpu_addrs, start=1):
        key = f"mpu{idx}"   # mpu1, mpu2 ...
        result[key] = read_one_mpu(addr)
    return result

# ===============================
# ③ SoftAP(Access Point) 구성
# ===============================
def connect():
    """
    Pico W를 WiFi AP(핫스팟) 모드로 활성화
    - SSID: PicoW
    - PASSWORD: 12345678
    """
    wlan = network.WLAN(network.AP_IF)
    wlan.active(False)
    wlan.config(ssid='PicoW', password='12345678')
    wlan.active(True)

    return wlan.ifconfig()[0]   # AP IP 주소 반환

# ===============================
# ④ 소켓 서버 생성
# ===============================
def open_socket():
    """
    포트 80에서 HTTP 요청을 수신하도록 소켓 오픈
    - SoftAP 상태의 PicoW가 간단한 서버 기능 수행
    """
    addr = ('0.0.0.0', 80)
    s = socket.socket()
    s.bind(addr)
    s.listen(2)
    s.settimeout(2)
    return s

# ===============================
# ⑤ 클라이언트 처리
# ===============================
def handle_client(connection):
    """
    클라이언트가 접속하면:
    - 요청 수신 (내용은 사용하지 않음)
    - 두 개의 MPU6050 센서 값을 JSON + 간단 HTTP 헤더로 전송
    """
    try:
        client, addr = connection.accept()
    except OSError:
        return

    try:
        client.settimeout(2)

        # 요청 데이터 읽기 (내용은 무시해도 됨)
        try:
            _ = client.recv(1024)
        except OSError:
            pass

        # 센서 데이터 읽기
        sensor_data = read_all_sensors()
        body = json.dumps(sensor_data)

        # 간단 HTTP 헤더 + JSON 바디
        header = (
            "HTTP/1.0 200 OK\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "\r\n"
        )

        client.sendall(header.encode() + body.encode())

    except OSError:
        pass
    finally:
        try:
            client.close()
        except OSError:
            pass

# ===============================
# ⑥ 메인 루프
# ===============================
try:
    ip = connect()
    print('AP IP:', ip)

    server = open_socket()
    print('Socket open')

    # 계속해서 연결 요청 처리 + 센서 값 콘솔 출력
    while True:
        # 콘솔에 두 센서 데이터 출력
        sensor_data = read_all_sensors()
        print("Sensor:", sensor_data)

        # 클라이언트가 접속하면 JSON 응답
        handle_client(server)

        time.sleep(0.1)

except KeyboardInterrupt:
    server.close()
    print('Server closed')
>>>>>>> 27539146d900344e26e343110250802df21560f6