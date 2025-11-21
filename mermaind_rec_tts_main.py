import cv2
import numpy as np
from ultralytics import YOLO
import requests
import math
import time
import pandas as pd
from datetime import datetime
import sys
from collections import deque
import pyttsx3
import threading
import queue

# ===============================
# 1. 설정값 (Thresholds)
# ===============================
EAR_SHOULDER_THRESH = 0.28  
SPINE_ANGLE_LIMIT = 35      
START_BENDING_THRESH = 15   
PELVIS_SENSOR_THRESH = 5.0  

SENSOR_URL = "http://192.168.4.1"
SENSOR_TIMEOUT = 0.05
WINDOW_NAME = "Mermaid AI Coach"
CALIBRATION_FRAMES = 60  # 캘리브레이션 프레임 수

# ===============================
# 2. TTS (음성) 설정
# ===============================
tts_queue = queue.Queue()

def tts_worker_thread():
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
        except Exception as e:
            print(f"TTS 오류 발생: {e}")
        tts_queue.task_done()

threading.Thread(target=tts_worker_thread, daemon=True).start()

def speak_message(message):
    if tts_queue.qsize() < 2: 
        # print(f"[🔊 음성 출력] {message}") 
        tts_queue.put(message)

last_speech_time = 0
speech_cooldown = 4.0 

def trigger_voice_feedback(message):
    global last_speech_time
    current_time = time.time()
    if current_time - last_speech_time > speech_cooldown:
        speak_message(message)
        last_speech_time = current_time

# ===============================
# 3. FPS 캘리브레이터 클래스 (추가됨)
# ===============================
class FPSCalibrator:
    """실제 처리 FPS 측정 및 캘리브레이션"""
    def __init__(self, calibration_frames=CALIBRATION_FRAMES):
        self.calibration_frames = calibration_frames
        self.frame_count = 0
        self.frame_times = deque(maxlen=calibration_frames)
        self.prev_time = time.perf_counter()
        self.is_calibrated = False
        self.measured_fps = 30.0
        self.measured_frame_interval = 1.0 / 30.0
    
    def update(self):
        current_time = time.perf_counter()
        if self.frame_count > 0:
            delta = current_time - self.prev_time
            if 0.01 < delta < 1.0:  # 이상값 제외
                self.frame_times.append(delta)
        self.prev_time = current_time
        self.frame_count += 1
        
        if not self.is_calibrated and len(self.frame_times) >= self.calibration_frames:
            self.finalize_calibration()
    
    def finalize_calibration(self):
        if len(self.frame_times) > 0:
            avg_time = np.mean(list(self.frame_times))
            self.measured_fps = 1.0 / avg_time if avg_time > 0 else 30.0
            self.measured_fps = max(5.0, min(60.0, self.measured_fps))
            self.measured_frame_interval = 1.0 / self.measured_fps
            self.is_calibrated = True
    
    def get_measured_fps(self):
        return self.measured_fps

    def get_progress(self):
        return int((len(self.frame_times) / self.calibration_frames) * 100)

# ===============================
# 4. 비디오 레코더 클래스 (추가됨)
# ===============================
class VideoRecorder:
    """실제 FPS로 저장하는 비디오 레코더"""
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = None
        self.frame_count = 0
        self.is_recording = False
        self.filename = None
        self.last_write_time = None
        self.frame_interval = 1.0 / fps
    
    def start_recording(self):
        try:
            self.filename = f"Mermaid_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (self.width, self.height))
            
            if not self.writer.isOpened():
                print("[ERROR] 비디오 라이터 초기화 실패")
                return False
            
            self.is_recording = True
            self.frame_count = 0
            self.last_write_time = time.perf_counter()
            print(f"\n[SUCCESS] 녹화 시작: {self.filename} (FPS: {self.fps:.2f})")
            return True
        except Exception as e:
            print(f"[ERROR] 녹화 시작 실패: {e}")
            return False
    
    def should_write_frame(self, current_time):
        if self.last_write_time is None: return True
        return (current_time - self.last_write_time) >= self.frame_interval
    
    def write_frame(self, frame, current_time):
        try:
            if self.writer and self.is_recording and self.should_write_frame(current_time):
                self.writer.write(frame)
                self.frame_count += 1
                self.last_write_time = current_time
                return True
        except Exception as e:
            print(f"[ERROR] 프레임 저장 실패: {e}")
        return False
    
    def stop_recording(self):
        try:
            if self.writer:
                self.writer.release()
                self.writer = None
                self.is_recording = False
                print(f"\n[SUCCESS] 녹화 완료: {self.filename} (Total Frames: {self.frame_count})")
                return True
        except Exception as e:
            print(f"[ERROR] 녹화 중지 실패: {e}")
        return False

# ===============================
# 5. YOLO 및 센서 로직
# ===============================
def calculate_mermaid_metrics(kps):
    l_sh, r_sh = kps[5][:2], kps[6][:2]
    l_ear, r_ear = kps[3][:2], kps[4][:2]
    l_hip, r_hip = kps[11][:2], kps[12][:2]

    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    torso_len = np.linalg.norm(mid_sh - mid_hip)
    
    if torso_len == 0: return 0, 0, 0

    spine_vec = mid_sh - mid_hip
    vertical_vec = np.array([0, -1])
    
    spine_u = spine_vec / np.linalg.norm(spine_vec)
    dot_prod = np.dot(spine_u, vertical_vec)
    spine_angle = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
    
    l_dist = np.linalg.norm(l_ear - l_sh) / torso_len
    r_dist = np.linalg.norm(r_ear - r_sh) / torso_len
    min_ear_sh_dist = min(l_dist, r_dist)

    return spine_angle, min_ear_sh_dist, torso_len

def check_basic_pose(kps):
    nose_conf = kps[0][2]
    hip_conf = min(kps[11][2], kps[12][2])
    if nose_conf < 0.5 or hip_conf < 0.5: return False
    return True

def get_mpu_data():
    try:
        r = requests.get(SENSOR_URL, timeout=SENSOR_TIMEOUT)
        if r.status_code == 200: return r.json()
    except: pass
    return None

def check_pelvis_sensor(sensor_data):
    if not sensor_data: return None, 0.0
    m1 = sensor_data.get("mpu1", {})
    m2 = sensor_data.get("mpu2", {})
    if not (m1 and m2): return None, 0.0

    def get_roll(m):
        ay = m.get("AcY", 0)
        az = m.get("AcZ", 1)
        return math.degrees(math.atan2(ay, az))

    r1 = get_roll(m1)
    r2 = get_roll(m2)
    diff = abs(r1 - r2)
    
    status = "LEVEL"
    if diff > PELVIS_SENSOR_THRESH: status = "UNSTABLE"
    return status, diff

# ===============================
# 6. 메인 실행 루프
# ===============================
def run_mermaid_coach():
    print("\n" + "="*60)
    print("  MERMAID AI COACH (Smart Recorder & Voice)")
    print("="*60 + "\n")

    # 변수 초기화
    recorded_data_log = []
    calibrator = FPSCalibrator(CALIBRATION_FRAMES)
    video_recorder = None
    
    # 윈도우 설정
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("[1/3] YOLO 모델 로딩 중...")
    model = YOLO("yolo11n-pose.pt") 

    print("[2/3] 웹캠 초기화 중...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[✗] 웹캠 열기 실패")
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[3/3] 준비 완료! (Resolution: {frame_width}x{frame_height})")
    speak_message("머메이드 코칭을 시작합니다. 캘리브레이션 중입니다.")

    frame_num = 0
    is_recording = False
    display_fps = 0.0
    recent_times = deque(maxlen=30)

    try:
        while True:
            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret: break
            
            frame_num += 1
            current_timestamp = time.time()
            h, w, _ = frame.shape

            # [FPS 캘리브레이션 업데이트]
            calibrator.update()

            # 1. YOLO 추론
            results = model(frame, verbose=False, conf=0.5)
            
            status_msg = "WAITING..."
            status_color = (200, 200, 200) 
            border_color = None
            
            log_entry = {'Timestamp': current_timestamp, 'Frame': frame_num}

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                frame = results[0].plot(img=frame)
                kps = results[0].keypoints.data[0].cpu().numpy()

                # 자세 분석 로직
                if check_basic_pose(kps):
                    spine_angle, es_dist, _ = calculate_mermaid_metrics(kps)
                    
                    if spine_angle < START_BENDING_THRESH:
                        # [READY]
                        status_msg = "START BENDING"
                        status_color = (0, 255, 255)
                        log_entry['State'] = 'Ready'
                    else:
                        # [EXERCISE]
                        sensor_data = get_mpu_data()
                        pelvis_status, pelvis_diff = check_pelvis_sensor(sensor_data)
                        
                        warning_list = []
                        
                        if pelvis_status == "UNSTABLE":
                            warning_list.append((f"FIX PELVIS! ({pelvis_diff:.1f})", "골반이 들리고 있어요"))
                        
                        if spine_angle > SPINE_ANGLE_LIMIT:
                            warning_list.append((f"LIMIT REACHED! ({spine_angle:.1f})", "너무 많이 꺾지 마세요"))

                        elif es_dist < EAR_SHOULDER_THRESH:
                            warning_list.append((f"SHOULDERS DOWN! ({es_dist:.2f})", "어깨를 내려주세요"))
                        
                        if not warning_list:
                            status_msg = f"GOOD ({spine_angle:.1f})"
                            status_color = (0, 255, 0)
                        else:
                            display_text, voice_text = warning_list[0]
                            status_msg = display_text
                            status_color = (0, 0, 255)
                            border_color = (0, 0, 255)
                            trigger_voice_feedback(voice_text)

                        log_entry.update({
                            'State': 'Exercise',
                            'Spine_Angle': spine_angle,
                            'Warning': warning_list[0][0] if warning_list else "None"
                        })
                else:
                    # [STANDBY]
                    status_msg = "SHOW FULL BODY & SIT UP"
                    status_color = (0, 0, 255)
                    log_entry['State'] = 'Standby'

                # 화면 텍스트 출력
                cv2.putText(frame, status_msg, (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
                if border_color:
                    cv2.rectangle(frame, (0,0), (w, h), border_color, 15)

            # ============ 정보 표시 (FPS, REC) ============
            # 캘리브레이션 상태 표시
            if not calibrator.is_calibrated:
                progress = calibrator.get_progress()
                cv2.putText(frame, f"Calibrating... {progress}%", (20, h - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            else:
                measured_fps = calibrator.get_measured_fps()
                cv2.putText(frame, f"Real FPS: {measured_fps:.1f}", (w - 250, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)

            # 녹화 상태 표시
            if is_recording:
                rec_text = f"REC ({video_recorder.frame_count if video_recorder else 0})"
                cv2.putText(frame, rec_text, (w - 250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                # 깜빡이는 빨간 점
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(frame, (w - 270, 80), 10, (0, 0, 255), -1)

            # ============ 녹화 및 화면 출력 ============
            
            # 프레임 저장 (FPS 캘리브레이션 적용)
            if is_recording and video_recorder:
                if video_recorder.write_frame(frame, current_timestamp):
                    if len(log_entry) > 2: recorded_data_log.append(log_entry)

            cv2.imshow(WINDOW_NAME, frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if is_recording:
                    is_recording = False
                    if video_recorder: video_recorder.stop_recording()
                else:
                    if not calibrator.is_calibrated:
                        print("[!] 캘리브레이션 중입니다. 잠시 후 다시 시도하세요.")
                    else:
                        is_recording = True
                        video_recorder = VideoRecorder(frame_width, frame_height, calibrator.get_measured_fps())
                        video_recorder.start_recording()

            # 디스플레이 FPS 계산
            loop_end = time.perf_counter()
            recent_times.append(loop_end - loop_start)
            display_fps = 1.0 / np.mean(list(recent_times)) if recent_times else 0

    except Exception as e:
        print(f"\n[ERROR] 실행 중 오류 발생: {e}")
    
    finally:
        cap.release()
        if video_recorder: video_recorder.stop_recording()
        cv2.destroyAllWindows()
        
        if recorded_data_log:
            pd.DataFrame(recorded_data_log).to_csv(f"Mermaid_Data_{int(time.time())}.csv", index=False)
            print("[✓] 데이터 로그 저장 완료")

if __name__ == "__main__":
    run_mermaid_coach()