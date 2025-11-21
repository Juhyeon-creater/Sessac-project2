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
from PIL import ImageFont, ImageDraw, Image

# ===============================
# 1. 설정값 (Thresholds)
# ===============================
EAR_SHOULDER_THRESH = 0.28
SPINE_ANGLE_LIMIT = 45  
START_BENDING_THRESH = 15   
PELVIS_SENSOR_THRESH = 5.0  

STABILITY_DURATION = 1.0    
MOVEMENT_THRESH = 0.04      

SENSOR_URL = "http://192.168.4.1"
SENSOR_TIMEOUT = 0.05
WINDOW_NAME = "Mermaid AI Coach (Smart Stability)"
CALIBRATION_FRAMES = 60

# ===============================
# 2. 유틸리티
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
        except Exception as e: print(f"TTS 오류: {e}")
        tts_queue.task_done()
threading.Thread(target=tts_worker_thread, daemon=True).start()

def speak_message(message):
    if tts_queue.qsize() < 2: tts_queue.put(message)

last_speech_time = 0
speech_cooldown = 4.0 
def trigger_voice_feedback(message):
    global last_speech_time
    current_time = time.time()
    if current_time - last_speech_time > speech_cooldown:
        speak_message(message)
        last_speech_time = current_time

class FPSCalibrator:
    def __init__(self, calibration_frames=CALIBRATION_FRAMES):
        self.calibration_frames = calibration_frames
        self.frame_count = 0
        self.frame_times = deque(maxlen=calibration_frames)
        self.prev_time = time.perf_counter()
        self.is_calibrated = False
        self.measured_fps = 30.0
    def update(self):
        current_time = time.perf_counter()
        if self.frame_count > 0:
            delta = current_time - self.prev_time
            if 0.01 < delta < 1.0: self.frame_times.append(delta)
        self.prev_time = current_time
        self.frame_count += 1
        if not self.is_calibrated and len(self.frame_times) >= self.calibration_frames:
            self.measured_fps = 1.0 / np.mean(list(self.frame_times))
            self.is_calibrated = True
    def get_measured_fps(self): return self.measured_fps
    def get_progress(self): return int((len(self.frame_times) / self.calibration_frames) * 100)

class VideoRecorder:
    def __init__(self, width, height, fps):
        self.writer = None; self.fps = fps; self.width = width; self.height = height
        self.is_recording = False; self.frame_interval = 1.0 / fps; self.last_write_time = None
    def start_recording(self):
        self.writer = cv2.VideoWriter(f"Mermaid_Log_{datetime.now().strftime('%H%M%S')}.mp4", 
                                      cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        self.is_recording = True; self.last_write_time = time.perf_counter()
    def write_frame(self, frame, current_time):
        if self.writer and self.is_recording:
            if self.last_write_time is None or (current_time - self.last_write_time) >= self.frame_interval:
                self.writer.write(frame)
                self.last_write_time = current_time
                return True
        return False
    def stop_recording(self):
        if self.writer: self.writer.release(); self.is_recording = False

# ===============================
# 3. YOLO 분석 및 스마트 감지
# ===============================
def calculate_mermaid_metrics(kps):
    l_sh, r_sh = kps[5][:2], kps[6][:2]
    l_ear, r_ear = kps[3][:2], kps[4][:2]
    l_hip, r_hip = kps[11][:2], kps[12][:2]
    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    torso_len = np.linalg.norm(mid_sh - mid_hip)
    if torso_len == 0: return 0, 0, 0, np.zeros((4, 2))
    spine_vec = mid_sh - mid_hip
    vertical_vec = np.array([0, -1])
    spine_u = spine_vec / np.linalg.norm(spine_vec)
    dot_prod = np.dot(spine_u, vertical_vec)
    spine_angle = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
    l_dist = np.linalg.norm(l_ear - l_sh) / torso_len
    r_dist = np.linalg.norm(r_ear - r_sh) / torso_len
    min_ear_sh_dist = min(l_dist, r_dist)
    current_pose = np.array([l_sh, r_sh, l_hip, r_hip])
    return spine_angle, min_ear_sh_dist, torso_len, current_pose

def calculate_movement(prev_pose, curr_pose, torso_len):
    if prev_pose is None: return 100.0 
    diff = np.linalg.norm(prev_pose - curr_pose, axis=1).sum()
    return diff / torso_len

def check_basic_pose(kps):
    nose_conf = kps[0][2]
    hip_conf = min(kps[11][2], kps[12][2])
    return nose_conf > 0.5 and hip_conf > 0.5

def check_floor_sitting(kps):
    l_hip_y, r_hip_y = kps[11][1], kps[12][1]
    l_knee_y, r_knee_y = kps[13][1], kps[14][1]
    hip_y_avg = (l_hip_y + r_hip_y) / 2
    knee_y_avg = (l_knee_y + r_knee_y) / 2
    mid_sh_y = (kps[5][1] + kps[6][1]) / 2
    torso_len = abs(hip_y_avg - mid_sh_y)
    leg_vertical_dist = knee_y_avg - hip_y_avg
    is_sitting = leg_vertical_dist < (torso_len * 0.6)
    return is_sitting

def get_mpu_data():
    try:
        r = requests.get(SENSOR_URL, timeout=SENSOR_TIMEOUT)
        if r.status_code == 200: return r.json()
    except: pass
    return None

def check_pelvis_sensor(sensor_data):
    if not sensor_data: return None, 0.0
    m1 = sensor_data.get("mpu1", {}); m2 = sensor_data.get("mpu2", {})
    if not (m1 and m2): return None, 0.0
    def get_roll(m): return math.degrees(math.atan2(m.get("AcY", 0), m.get("AcZ", 1)))
    diff = abs(get_roll(m1) - get_roll(m2))
    return ("UNSTABLE" if diff > PELVIS_SENSOR_THRESH else "LEVEL"), diff

# ===============================
# 4. 메인 실행 루프
# ===============================
def run_mermaid_coach():
    print("--- MERMAID AI COACH (Stability Locked) ---")
    
    recorded_data_log = []
    calibrator = FPSCalibrator()
    video_recorder = None
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("Loading YOLO...")
    model = YOLO("yolo11n-pose.pt") 
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    speak_message("머메이드 코칭을 시작합니다. 바닥에 편하게 앉아주세요.")

    frame_num = 0
    is_recording = False
    
    pose_history = deque(maxlen=5) 
    prev_smooth_pose = None
    stable_frame_count = 0
    REQUIRED_STABLE_FRAMES = 30 
    
    # [핵심] 상태 기억 변수
    is_stabilized = False  # 한번 준비완료 되면 True로 고정

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_num += 1; current_timestamp = time.time(); h, w, _ = frame.shape
            calibrator.update()

            results = model(frame, verbose=False, conf=0.5)
            status_msg = "대기 중..."; status_color = (200, 200, 200); border_color = None
            log_entry = {'Timestamp': current_timestamp, 'Frame': frame_num}

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                frame = results[0].plot(img=frame)
                kps = results[0].keypoints.data[0].cpu().numpy()

                if check_basic_pose(kps):
                    is_sitting = check_floor_sitting(kps)
                    if is_sitting:
                        spine_angle, es_dist, torso_len, curr_pose_coords = calculate_mermaid_metrics(kps)
                        
                        # --- 스무딩 및 움직임 계산 ---
                        pose_history.append(curr_pose_coords)
                        if len(pose_history) == 5:
                            smooth_pose = np.mean(pose_history, axis=0)
                            movement_score = calculate_movement(prev_smooth_pose, smooth_pose, torso_len)
                            prev_smooth_pose = smooth_pose 
                            
                            if movement_score < MOVEMENT_THRESH:
                                stable_frame_count += 1
                            else:
                                stable_frame_count = 0 
                        else:
                            movement_score = 100.0

                        # --- 상태 판정 ---
                        if spine_angle < START_BENDING_THRESH:
                            # [READY] 
                            
                            # 이미 안정화된 상태라면(is_stabilized=True) 게이지 로직 건너뜀
                            if is_stabilized:
                                status_msg = "준비 완료! 옆으로 숙이세요"
                                status_color = (0, 255, 0) 
                                log_entry['State'] = 'Ready_Locked'
                            else:
                                # 아직 안정화 안 됨 -> 게이지 채우기
                                progress = min(stable_frame_count / REQUIRED_STABLE_FRAMES, 1.0)
                                
                                if progress < 1.0:
                                    status_msg = f"움직이지 마세요... {int(progress*100)}%"
                                    status_color = (0, 255, 255) 
                                else:
                                    # 100% 달성 -> 상태 잠금! (Lock)
                                    is_stabilized = True
                                    status_msg = "준비 완료! 옆으로 숙이세요"
                                    status_color = (0, 255, 0) 
                                    speak_message("준비 자세가 확인되었습니다. 머메이드 동작을 시작하세요.")
                                    log_entry['State'] = 'Ready'

                                # 게이지 바 표시 (아직 잠금 전일 때만)
                                if not is_stabilized:
                                    bar_w = int(w * 0.6 * progress)
                                    cv2.rectangle(frame, (int(w*0.2), h-80), (int(w*0.2)+bar_w, h-60), (0, 255, 0), -1)
                                    cv2.rectangle(frame, (int(w*0.2), h-80), (int(w*0.2)+int(w*0.6), h-60), (255, 255, 255), 2)

                        else:
                            # [EXERCISE]
                            # 운동 시작하면 당연히 안정화 완료된 것으로 간주 (혹시 바로 굽혔을 경우 대비)
                            is_stabilized = True 
                            
                            log_entry['State'] = 'Exercise'
                            sensor_data = get_mpu_data()
                            pelvis_status, pelvis_diff = check_pelvis_sensor(sensor_data)
                            
                            warning_list = []
                            if pelvis_status == "UNSTABLE": warning_list.append((f"골반비틀림! ({pelvis_diff:.1f})", "골반이 들리고 있어요"))
                            if es_dist < EAR_SHOULDER_THRESH: warning_list.append((f"어깨 내리세요! ({es_dist:.2f})", "어깨를 내려주세요"))
                            if spine_angle > SPINE_ANGLE_LIMIT: warning_list.append((f"무리한 꺾기! ({spine_angle:.1f})", "너무 많이 꺾지 마세요"))
                            
                            if not warning_list:
                                status_msg = f"좋음 ({spine_angle:.1f})"
                                status_color = (0, 255, 0)
                            else:
                                display_text, voice_text = warning_list[0]
                                status_msg = display_text
                                status_color = (0, 0, 255); border_color = (0, 0, 255)
                                trigger_voice_feedback(voice_text)

                            log_entry.update({'Spine_Angle': spine_angle, 'Warning': warning_list[0][0] if warning_list else "None"})
                    else:
                        status_msg = "바닥에 앉아주세요"
                        status_color = (0, 165, 255)
                        log_entry['State'] = 'Standby_Pose_Error'
                        # 자세 풀리면 리셋
                        is_stabilized = False 
                        stable_frame_count = 0

                else:
                    status_msg = "전신이 보이게 앉아주세요"
                    status_color = (0, 0, 255)
                    log_entry['State'] = 'Standby'
                    # 전신 안 보이면 리셋
                    is_stabilized = False
                    stable_frame_count = 0
                    pose_history.clear()

                frame = draw_korean_text(frame, status_msg, (50, 100), 40, status_color)
                if border_color: cv2.rectangle(frame, (0,0), (w, h), border_color, 15)

            # FPS 및 녹화 처리
            if not calibrator.is_calibrated:
                progress = calibrator.get_progress()
                cv2.putText(frame, f"Calibrating... {progress}%", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            else:
                measured_fps = calibrator.get_measured_fps()
                cv2.putText(frame, f"Real FPS: {measured_fps:.1f}", (w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)

            if is_recording:
                rec_text = f"REC ({video_recorder.frame_count if video_recorder else 0})"
                cv2.putText(frame, rec_text, (w - 250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                if int(time.time() * 2) % 2 == 0: cv2.circle(frame, (w - 270, 80), 10, (0, 0, 255), -1)

            if is_recording and video_recorder:
                if video_recorder.write_frame(frame, current_timestamp):
                    if len(log_entry) > 2: recorded_data_log.append(log_entry)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                if is_recording: 
                    is_recording = False
                    if video_recorder: video_recorder.stop_recording()
                else:
                    if not calibrator.is_calibrated: print("[!] 캘리브레이션 중입니다.")
                    else:
                        is_recording = True
                        video_recorder = VideoRecorder(frame_width, frame_height, calibrator.get_measured_fps())
                        video_recorder.start_recording()

    except Exception as e: print(f"Error: {e}")
    finally:
        cap.release()
        if video_recorder: video_recorder.stop_recording()
        cv2.destroyAllWindows()
        if recorded_data_log: pd.DataFrame(recorded_data_log).to_csv("Mermaid_Log.csv", index=False)

if __name__ == "__main__":
    run_mermaid_coach()