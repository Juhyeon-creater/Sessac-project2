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
import os
from PIL import ImageFont, ImageDraw, Image

# ===============================
# 1. 설정값 (Thresholds)
# ===============================
EAR_SHOULDER_THRESH = 0.28  
SPINE_ANGLE_LIMIT = 45      
START_BENDING_THRESH = 15   
STRICT_READY_ANGLE = 10.0   
SHOULDER_LEVEL_THRESH = 0.1 
PELVIS_SENSOR_THRESH = 5.0  

STABILITY_DURATION = 1.0    
MOVEMENT_THRESH = 0.04      

SENSOR_URL = "http://192.168.4.1"
SENSOR_TIMEOUT = 0.05
WINDOW_NAME = "Mermaid AI Coach (Voice Mode)"
CALIBRATION_FRAMES = 60

GUIDE_IMAGE_PATH = "mermaid/reference_pose.png" 

# ===============================
# 2. 디자인 유틸리티 (UI/UX)
# ===============================
def get_font(size):
    try: return ImageFont.truetype("malgun.ttf", size)
    except: return ImageFont.load_default()

def draw_ui_text(img, text, pos, font_size, bg_color=(0,0,0), text_color=(255,255,255), align="left"):
    """
    반투명 배경이 있는 예쁜 텍스트 그리기
    """
    if not text: return img 

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, 'RGBA')
    font = get_font(font_size)
    
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    x, y = pos
    padding_x = 12
    padding_y = 4
    
    if align == "center":
        x = x - (w // 2)
        bg_box = [x - padding_x, y - padding_y, x + w + padding_x, y + h + padding_y + 2]
    else:
        bg_box = [x, y, x + w + padding_x * 2, y + h + padding_y * 2]
        x += padding_x
        y += padding_y

    r, g, b = bg_color
    draw.rectangle(bg_box, fill=(r, g, b, 153)) 
    draw.text((x, y), text, font=font, fill=(*text_color, 255))
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ===============================
# 3. TTS & Recorders
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
            if delta > 0.001: 
                self.frame_times.append(delta)
        self.prev_time = current_time
        self.frame_count += 1
        
        if not self.is_calibrated and len(self.frame_times) >= self.calibration_frames:
            self.measured_fps = 1.0 / np.mean(list(self.frame_times))
            self.is_calibrated = True
            
    def get_best_fps(self):
        if self.is_calibrated:
            return self.measured_fps
        elif len(self.frame_times) > 5:
            return 1.0 / np.mean(list(self.frame_times))
        else:
            return 30.0

class VideoRecorder:
    def __init__(self, width, height, fps):
        self.writer = None; self.fps = fps; self.width = width; self.height = height
        self.is_recording = False; self.frame_interval = 1.0 / fps; self.last_write_time = None
    def start_recording(self):
        filename = f"Mermaid_Video_{datetime.now().strftime('%H%M%S')}.mp4"
        print(f"[녹화 시작] 파일명: {filename}, FPS: {self.fps:.2f}")
        self.writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        self.is_recording = True; self.last_write_time = time.perf_counter()
    def write_frame(self, frame, current_time):
        if self.writer and self.is_recording:
            if self.last_write_time is None or (current_time - self.last_write_time) >= self.frame_interval:
                self.writer.write(frame)
                self.last_write_time = current_time
                return True
        return False
    def stop_recording(self):
        if self.writer: 
            self.writer.release()
            print("[녹화 종료] 파일 저장 완료")
        self.is_recording = False

# ===============================
# 4. YOLO 분석 로직
# ===============================
def calculate_mermaid_metrics(kps):
    l_sh, r_sh = kps[5][:2], kps[6][:2]
    l_ear, r_ear = kps[3][:2], kps[4][:2]
    l_hip, r_hip = kps[11][:2], kps[12][:2]
    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    torso_len = np.linalg.norm(mid_sh - mid_hip)
    if torso_len == 0: return 0, 0, 0, np.zeros((4, 2)), 0
    spine_vec = mid_sh - mid_hip
    vertical_vec = np.array([0, -1])
    spine_u = spine_vec / np.linalg.norm(spine_vec)
    dot_prod = np.dot(spine_u, vertical_vec)
    spine_angle = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
    l_dist = np.linalg.norm(l_ear - l_sh) / torso_len
    r_dist = np.linalg.norm(r_ear - r_sh) / torso_len
    min_ear_sh_dist = min(l_dist, r_dist)
    current_pose = np.array([l_sh, r_sh, l_hip, r_hip])
    shoulder_level = abs(l_sh[1] - r_sh[1]) / torso_len
    return spine_angle, min_ear_sh_dist, torso_len, current_pose, shoulder_level

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
    if not sensor_data: return "LEVEL", 0.0
    m1 = sensor_data.get("mpu1", {}); m2 = sensor_data.get("mpu2", {})
    if not (m1 and m2): return "LEVEL", 0.0
    def get_roll(m): return math.degrees(math.atan2(m.get("AcY", 0), m.get("AcZ", 1)))
    r1 = get_roll(m1); r2 = get_roll(m2)
    diff = r1 - r2 
    if abs(diff) <= PELVIS_SENSOR_THRESH: return "LEVEL", abs(diff)
    elif diff > PELVIS_SENSOR_THRESH: return "LEFT_UP", abs(diff)
    else: return "RIGHT_UP", abs(diff)

# ===============================
# 5. 메인 실행 루프
# ===============================
def run_mermaid_coach():
    print("--- MERMAID AI COACH (V12: Continuous Logging) ---")
    
    guide_img_original = None
    if os.path.exists(GUIDE_IMAGE_PATH):
        guide_img_original = cv2.imread(GUIDE_IMAGE_PATH)
        if guide_img_original is None: 
            temp = cv2.imread(GUIDE_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
            if temp is not None and temp.shape[2] == 4:
                trans_mask = temp[:,:,3] == 0
                temp[trans_mask] = [255, 255, 255, 0]
                guide_img_original = cv2.cvtColor(temp, cv2.COLOR_BGRA2BGR)
            else: guide_img_original = temp
        print(f"[✓] 가이드 이미지 로드 성공")

    recorded_data_log = []
    calibrator = FPSCalibrator()
    video_recorder = None
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 480) 

    print("Loading YOLO...")
    model = YOLO("yolo11n-pose.pt") 
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    guide_img_resized = None
    if guide_img_original is not None:
        h_guide, w_guide = guide_img_original.shape[:2]
        scale = frame_height / h_guide
        guide_img_resized = cv2.resize(guide_img_original, (int(w_guide * scale), frame_height))

    speak_message("전신이 보이게 앉아주세요.")

    frame_num = 0
    is_recording = False
    
    current_speech_state = "INIT" 
    new_speech_state = "INIT"
    has_said_start = False 
    
    pose_history = deque(maxlen=5) 
    prev_smooth_pose = None
    stable_frame_count = 0
    REQUIRED_STABLE_FRAMES = 30 
    is_stabilized = False  

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame_num += 1; current_timestamp = time.time(); h, w, _ = frame.shape
            calibrator.update()

            new_speech_state = current_speech_state 
            results = model(frame, verbose=False, conf=0.5)
            log_entry = {'Timestamp': current_timestamp, 'Frame': frame_num}
            
            badge_text = "대기 중"
            badge_color = (100, 100, 100) 
            subtitle_text = "" 
            border_color = None

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                frame = results[0].plot(img=frame)
                kps = results[0].keypoints.data[0].cpu().numpy()

                if check_basic_pose(kps):
                    is_sitting = check_floor_sitting(kps)
                    if is_sitting:
                        # [1] 앉아 있으면 무조건 모든 데이터 수집 (상태 무관)
                        spine_angle, es_dist, torso_len, curr_pose_coords, shoulder_level = calculate_mermaid_metrics(kps)
                        sensor_data = get_mpu_data() # 센서값도 항상 가져옴
                        pelvis_status, pelvis_diff = check_pelvis_sensor(sensor_data)
                        
                        # [2] 로그에 주요 수치 기록 (이 시점에서 미리 기록)
                        log_entry.update({
                            'Spine_Angle': spine_angle, 
                            'Ear_Shoulder_Dist': es_dist, # 귀-어깨 거리 추가
                            'Pelvis_Tilt': pelvis_diff,
                            'Torso_Length': torso_len,
                            'Warning': "None" # 기본값
                        })
                        
                        pose_history.append(curr_pose_coords)
                        if len(pose_history) == 5:
                            smooth_pose = np.mean(pose_history, axis=0)
                            movement_score = calculate_movement(prev_smooth_pose, smooth_pose, torso_len)
                            prev_smooth_pose = smooth_pose 
                            if movement_score < MOVEMENT_THRESH: stable_frame_count += 1
                            else: stable_frame_count = 0 
                        else: movement_score = 100.0

                        if spine_angle < START_BENDING_THRESH:
                            # [READY 단계]
                            log_entry['State'] = 'Ready' # 상태 기록
                            if is_stabilized:
                                badge_text = "준비 완료"
                                badge_color = (0, 180, 0)
                                subtitle_text = "옆으로 천천히 숙이세요" 
                                log_entry['State'] = 'Ready_Locked'
                                new_speech_state = "READY_COMPLETE"
                            else:
                                is_strict_ready = (spine_angle < STRICT_READY_ANGLE) and (shoulder_level < SHOULDER_LEVEL_THRESH)
                                if not is_strict_ready:
                                    badge_text = "자세 교정 필요"
                                    badge_color = (0, 165, 255)
                                    subtitle_text = "척추와 어깨를 바르게 펴세요" 
                                    stable_frame_count = 0
                                else:
                                    progress = min(stable_frame_count / REQUIRED_STABLE_FRAMES, 1.0)
                                    if progress < 1.0:
                                        badge_text = "안정화 중..."
                                        badge_color = (0, 165, 255) 
                                        subtitle_text = f"움직이지 마세요 {int(progress*100)}%" 
                                        new_speech_state = "STABILIZING"
                                    else:
                                        is_stabilized = True
                                        new_speech_state = "READY_COMPLETE"

                                if not is_stabilized and is_strict_ready:
                                    bar_w = int(w * 0.6 * progress)
                                    cv2.rectangle(frame, (int(w*0.2), h-25), (int(w*0.2)+bar_w, h-15), (0, 255, 0), -1)
                                    cv2.rectangle(frame, (int(w*0.2), h-25), (int(w*0.2)+int(w*0.6), h-15), (200, 200, 200), 1)

                        else:
                            # [EXERCISE 단계]
                            is_stabilized = True 
                            new_speech_state = "EXERCISE"
                            log_entry['State'] = 'Exercise'
                            
                            warning_list = []
                            if pelvis_status == "LEFT_UP": 
                                warning_list.append(("Left Pelvis Lift", "왼쪽 골반을 내려주세요"))
                            elif pelvis_status == "RIGHT_UP": 
                                warning_list.append(("Right Pelvis Lift", "오른쪽 골반을 내려주세요"))
                            if es_dist < EAR_SHOULDER_THRESH: 
                                warning_list.append(("Shoulder Shrug", "어깨를 내려주세요"))
                            if spine_angle > SPINE_ANGLE_LIMIT: 
                                warning_list.append(("Excessive Bending", "너무 많이 꺾지 마세요"))
                            
                            if not warning_list:
                                badge_text = f"자세 정확 ({int(spine_angle)}°)"
                                badge_color = (0, 180, 0)
                                subtitle_text = "" 
                            else:
                                display_text, voice_text = warning_list[0]
                                badge_text = "Warning" 
                                badge_color = (200, 0, 0)
                                subtitle_text = "" 
                                border_color = (0, 0, 255)
                                trigger_voice_feedback(voice_text)
                                # 경고 발생 시 로그 업데이트
                                log_entry['Warning'] = warning_list[0][0]

                    else:
                        badge_text = "대기 중"
                        subtitle_text = "바닥에 편하게 앉아주세요" 
                        log_entry['State'] = 'Standby_Pose_Error'
                        is_stabilized = False 
                        stable_frame_count = 0
                        new_speech_state = "STANDBY"
                else:
                    badge_text = "인식 불가"
                    subtitle_text = "전신이 보이게 앉아주세요" 
                    log_entry['State'] = 'Standby'
                    is_stabilized = False
                    stable_frame_count = 0
                    pose_history.clear()
                    new_speech_state = "STANDBY"

                frame = draw_ui_text(frame, badge_text, (20, 20), 24, bg_color=badge_color, align="left")
                if subtitle_text:
                    frame = draw_ui_text(frame, subtitle_text, (w//2, h-50), 20, bg_color=(0,0,0), align="center")
                if border_color:
                    cv2.rectangle(frame, (0,0), (w, h), border_color, 15)
            
            if new_speech_state != "INIT" and new_speech_state != current_speech_state:
                if new_speech_state == "STANDBY": 
                    speak_message("전신이 보이게 앉아주세요.")
                elif new_speech_state == "STABILIZING": 
                    pass 
                elif new_speech_state == "READY_COMPLETE": 
                    if not has_said_start: 
                        speak_message("AI 코칭을 시작합니다.")
                        has_said_start = True
                current_speech_state = new_speech_state
            
            final_display = frame
            if guide_img_resized is not None:
                final_display = np.hstack((guide_img_resized, frame))
                cv2.putText(final_display, "GUIDE POSE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            output_w = 1280
            aspect_ratio = final_display.shape[0] / final_display.shape[1]
            output_h = int(output_w * aspect_ratio)
            final_display_resized = cv2.resize(final_display, (output_w, output_h))
            
            if is_recording:
                cv2.circle(final_display_resized, (output_w - 30, 30), 10, (0, 0, 255), -1)

            cv2.imshow(WINDOW_NAME, final_display_resized)
            
            if is_recording and video_recorder:
                video_recorder.write_frame(frame, current_timestamp)
                # [중요] 녹화 중일 때만 데이터 수집 (단, 모든 상태 기록 포함)
                if len(log_entry) > 2: 
                    recorded_data_log.append(log_entry)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                if is_recording: 
                    is_recording = False
                    if video_recorder: video_recorder.stop_recording()
                else:
                    fps = calibrator.get_best_fps()
                    is_recording = True
                    video_recorder = VideoRecorder(frame_width, frame_height, fps)
                    video_recorder.start_recording()

    except Exception as e: print(f"Error: {e}")
    finally:
        cap.release()
        if video_recorder: video_recorder.stop_recording()
        cv2.destroyAllWindows()
        
        if recorded_data_log:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f"Mermaid_Log_{timestamp_str}.csv"
            pd.DataFrame(recorded_data_log).to_csv(csv_filename, index=False)
            print(f"[저장 완료] 로그 파일: {csv_filename}")

if __name__ == "__main__":
    run_mermaid_coach()