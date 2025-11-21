import cv2
import numpy as np
from ultralytics import YOLO
import requests
import math
import time
import pandas as pd
from datetime import datetime
import pyttsx3
import threading
import queue
from PIL import ImageFont, ImageDraw, Image

# ===============================
# 1. 설정값 (Thresholds)
# ===============================
# 런지 각도 기준 (90도가 이상적)
LUNGE_TARGET_ANGLE = 90
ANGLE_TOLERANCE_GOOD = 15  # 75 ~ 105도 (Good)
ANGLE_TOLERANCE_WARN = 25  # 65 ~ 115도 (Warning)

# 측면 감지 기준 (어깨 너비 비율)
SIDE_VIEW_RATIO_THRESH = 0.45 

# 골반 센서 기준
PELVIS_SENSOR_THRESH = 5.0 
SENSOR_URL = "http://192.168.4.1"
SENSOR_TIMEOUT = 0.05

WINDOW_NAME = "Lunge Assessment"

# ===============================
# 2. 유틸리티 (한글, 음성, 녹화)
# ===============================
def draw_korean_text(img, text, position, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("malgun.ttf", font_size)
    except: font = ImageFont.load_default()
    fill_color = (color[2], color[1], color[0]) 
    draw.text(position, text, font=font, fill=fill_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

tts_queue = queue.Queue()
def tts_worker():
    while True:
        msg = tts_queue.get()
        if msg is None: break
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(msg)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e: print(f"TTS Error: {e}")
        tts_queue.task_done()
threading.Thread(target=tts_worker, daemon=True).start()

def speak_message(message):
    if tts_queue.qsize() < 2: tts_queue.put(message)

last_speech_time = 0
speech_cooldown = 4.0 
def trigger_voice_feedback(message):
    global last_speech_time
    if time.time() - last_speech_time > speech_cooldown:
        speak_message(message)
        last_speech_time = time.time()

# ===============================
# 3. 런지 분석 로직
# ===============================
def check_side_view(kps):
    """측면 여부 확인 (어깨 너비 비율)"""
    l_sh, r_sh = kps[5][:2], kps[6][:2]
    l_hip, r_hip = kps[11][:2], kps[12][:2]
    
    shoulder_width = abs(l_sh[0] - r_sh[0])
    mid_sh_y = (l_sh[1] + r_sh[1]) / 2
    mid_hip_y = (l_hip[1] + r_hip[1]) / 2
    torso_height = abs(mid_sh_y - mid_hip_y)
    
    if torso_height == 0: return False
    ratio = shoulder_width / torso_height
    return ratio < SIDE_VIEW_RATIO_THRESH

def calculate_lunge_metrics(kps):
    """런지 무릎 각도 및 점수 계산"""
    # 앞다리 자동 감지 (더 많이 굽혀진 쪽)
    def get_knee_angle(hip, knee, ankle):
        ba = hip - knee
        bc = ankle - knee
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    l_angle = get_knee_angle(kps[11][:2], kps[13][:2], kps[15][:2])
    r_angle = get_knee_angle(kps[12][:2], kps[14][:2], kps[16][:2])

    if l_angle < r_angle:
        front_angle = l_angle
        working_leg = "Left"
    else:
        front_angle = r_angle
        working_leg = "Right"
    
    # 점수 계산 (100점 만점 감점제)
    # 목표 각도(90도)와의 차이만큼 감점
    angle_diff = abs(LUNGE_TARGET_ANGLE - front_angle)
    score = max(0, 100 - (angle_diff * 1.5)) # 1도 차이당 1.5점 감점
    
    return front_angle, int(score), working_leg

def get_mpu_data():
    try:
        r = requests.get(SENSOR_URL, timeout=SENSOR_TIMEOUT)
        if r.status_code == 200: return r.json()
    except: pass
    return None

def check_pelvis_sensor(sensor_data):
    if not sensor_data: return "LEVEL", 0.0
    m1 = sensor_data.get("mpu1", {})
    m2 = sensor_data.get("mpu2", {})
    if not (m1 and m2): return "LEVEL", 0.0
    
    def get_roll(m): return math.degrees(math.atan2(m.get("AcY", 0), m.get("AcZ", 1)))
    r1 = get_roll(m1); r2 = get_roll(m2)
    diff = r1 - r2
    
    if abs(diff) <= PELVIS_SENSOR_THRESH: return "LEVEL", abs(diff)
    elif diff > PELVIS_SENSOR_THRESH: return "LEFT_UP", abs(diff)
    else: return "RIGHT_UP", abs(diff)

# ===============================
# 4. 메인 실행 루프
# ===============================
def run_lunge_assessment():
    print("--- LUNGE ASSESSMENT STARTED ---")
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("Loading YOLO...")
    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(0)
    
    # 초기 상태
    current_state = "INIT" 
    speak_message("신체평가를 시작합니다. 카메라를 향해 옆으로 서주세요.")
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) # 거울 모드
        frame_count += 1
        
        results = model(frame, verbose=False, conf=0.5)
        
        # UI 기본값
        status_text = "대기 중"
        status_color = (200, 200, 200)
        feedback_text = ""
        
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            frame = results[0].plot(img=frame)
            kps = results[0].keypoints.data[0].cpu().numpy()
            
            # [Phase 1] 측면 확인 (Side View Check)
            is_side = check_side_view(kps)
            
            if current_state == "INIT":
                if is_side:
                    current_state = "READY"
                    speak_message("측면이 확인되었습니다. 한쪽 발을 앞으로 뻗어 런지 준비 자세를 취하세요.")
                else:
                    status_text = "옆으로 서주세요"
                    status_color = (0, 0, 255)
                    if frame_count % 90 == 0: # 3초마다 안내
                        speak_message("카메라를 향해 옆으로 서주세요.")

            # [Phase 2 & 3] 준비 및 운동 (Ready & Exercise)
            elif current_state in ["READY", "EXERCISE"]:
                front_angle, score, leg = calculate_lunge_metrics(kps)
                
                # 준비 자세 판단 (다리를 벌리고 섰는지 -> 각도 150도 미만이면 시작으로 간주)
                if current_state == "READY":
                    status_text = "준비 완료"
                    status_color = (0, 255, 255)
                    feedback_text = "준비 자세. 천천히 내려가세요."
                    
                    if front_angle < 150: # 무릎 굽히기 시작
                        current_state = "EXERCISE"
                        speak_message("런지 시작. 무릎 각도에 주의하세요.")
                
                else: # EXERCISE
                    # 센서 확인
                    sensor_data = get_mpu_data()
                    pelvis_status, pelvis_diff = check_pelvis_sensor(sensor_data)
                    
                    # 피드백 로직
                    warning_msg = ""
                    
                    # 1. 골반 체크
                    if pelvis_status == "LEFT_UP":
                        warning_msg = "왼쪽 골반 비틀림!"
                        trigger_voice_feedback("왼쪽 골반이 들렸습니다.")
                    elif pelvis_status == "RIGHT_UP":
                        warning_msg = "오른쪽 골반 비틀림!"
                        trigger_voice_feedback("오른쪽 골반이 들렸습니다.")
                    
                    # 2. 무릎 각도 체크
                    elif front_angle < (LUNGE_TARGET_ANGLE - ANGLE_TOLERANCE_WARN): # < 65도 (너무 깊음/앞으로 쏠림)
                        warning_msg = "무릎이 발끝을 넘습니다!"
                        trigger_voice_feedback("무릎이 발끝을 넘지 않게 하세요.")
                    elif front_angle > (LUNGE_TARGET_ANGLE + ANGLE_TOLERANCE_WARN): # > 115도 (너무 얕음)
                        warning_msg = "더 깊게 앉으세요"
                        if front_angle < 140: # 완전히 서있을 때 말고, 운동 중에만
                            trigger_voice_feedback("더 깊게 앉으세요.")
                    else:
                        # Good
                        warning_msg = "좋습니다!"
                        
                    # 상태 표시
                    status_text = f"점수: {score}점 ({int(front_angle)}°)"
                    status_color = (0, 255, 0) if score > 80 else (0, 165, 255)
                    feedback_text = warning_msg

        # 화면 출력
        frame = draw_korean_text(frame, status_text, (20, 50), 30, status_color)
        if feedback_text:
            frame = draw_korean_text(frame, feedback_text, (20, 100), 40, (255, 255, 255)) # 흰색 자막

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_lunge_assessment()