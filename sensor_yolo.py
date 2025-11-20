import cv2
import numpy as np
import threading
import time
import socket
import json
from ultralytics import YOLO

# --- 1. 전역 설정 및 변수 ---

# 사용자 지정 규칙 적용: 실시간용은 yolo11n 사용
MODEL_NAME = 'yolo11n-pose.pt' 
PICO_W_IP = '192.168.4.1' # Pico W SoftAP의 기본 IP 주소
PICO_W_PORT = 80          # Pico W 서버 포트

# 스레드 간 데이터 공유를 위한 변수
global_mpu_data = {"mpu1": {"GyX": 0, "GyY": 0, "GyZ": 0}, "mpu2": {"GyX": 0, "GyY": 0, "GyZ": 0}}
data_lock = threading.Lock() # 데이터 안전 공유를 위한 락(Lock)

# --- 2. MPU 데이터 수신 스레드 ---
def sensor_client_thread():
    """ Pico W 서버에 접속하여 센서 데이터를 지속적으로 받는 클라이언트 스레드 """
    global global_mpu_data
    
    # HTTP 요청 헤더 (Pico W 서버는 간단한 HTTP 응답을 하므로 요청도 HTTP 형식으로)
    request_header = f"GET / HTTP/1.1\r\nHost: {PICO_W_IP}\r\nConnection: close\r\n\r\n".encode()

    while True:
        try:
            # 1. 소켓 연결 (Pico W는 TCP 서버)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3) # 연결 대기 시간 3초
            s.connect((PICO_W_IP, PICO_W_PORT))
            
            # 2. 데이터 요청
            s.sendall(request_header)
            
            # 3. 응답 수신
            response = b''
            while True:
                chunk = s.recv(1024)
                if not chunk:
                    break
                response += chunk
            
            # 4. JSON 파싱
            if response:
                # HTTP 헤더 제거 후 JSON만 파싱
                _, body = response.split(b'\r\n\r\n', 1) 
                
                new_data = json.loads(body.decode('utf-8'))
                
                # 5. [LOCK] 데이터 안전하게 업데이트
                with data_lock:
                    global_mpu_data = new_data
            
            s.close()
            time.sleep(0.05) # 너무 빠른 폴링 방지 (20ms)

        except (socket.error, ConnectionRefusedError, json.JSONDecodeError, OSError) as e:
            # print(f"❌ MPU Client Error: {e}") 
            time.sleep(1) # 연결 실패 시 1초 대기 후 재시도

# --- 3. 메인 YOLO 처리 루프 (Main Thread) ---
def main_yolo_loop():
    
    # YOLO 모델 로드
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(0)

    # YOLO 추론 및 기타 로직 (다리 각도 등)은 생략하고 융합 표시만 구현
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 1. YOLO 추론 (이전 코드의 다리 각도 계산 로직이 여기에 들어갑니다)
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # 2. [LOCK] 공유 변수에서 MPU 데이터 안전하게 읽기
        mpu_display_data = None
        with data_lock:
            mpu_display_data = global_mpu_data.copy()

        # 3. 화면 융합 (MPU 데이터 표시)
        if mpu_display_data:
            gy_x1 = mpu_display_data.get('mpu1', {}).get('GyX', 0)
            gy_x2 = mpu_display_data.get('mpu2', {}).get('GyX', 0)

            # 예시: 두 센서의 GyX 평균으로 골반 기울기 추정 (YOLO 각도와 함께 표시)
            fused_value = (gy_x1 + gy_x2) / 2 
            
            cv2.putText(annotated_frame, f"YOLO Angle: {90.0}", (30, 50), # (여기에 실제 YOLO 각도 값 대입)
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"MPU Fused Tilt: {fused_value:.0f}", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("YOLO + MPU Fusion System", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("YOLO Loop Closed.")

# --- 4. 시스템 시작 ---
if __name__ == '__main__':
    # 1. MPU 데이터 수신 스레드 시작
    mpu_thread = threading.Thread(target=sensor_client_thread, daemon=True)
    mpu_thread.start()
    print(f"MPU Client Thread Started, waiting for Pico W at {PICO_W_IP}:{PICO_W_PORT}")
    
    # 2. 메인 YOLO 루프 시작 (비디오 및 화면 제어)
    try:
        main_yolo_loop()
    except KeyboardInterrupt:
        pass
    
    print("System Shutdown Complete.")