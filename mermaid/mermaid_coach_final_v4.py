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
WINDOW_NAME = "Mermaid AI Coach (Design Ver)"
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
    반투명 배경이 있는 예쁜 텍스트 그리기 (OpenCV 이미지에 적용)
    align: "left"(상태 배지용), "center"(피드백 자막용)
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, 'RGBA')
    font = get_font(font_size)
    
    # 텍스트 크기 측정
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    x, y = pos
    padding = 15
    
    # 위치 및 배경 박스 계산
    if align == "center":
        x = x - (w // 2)
        bg_box = [x - padding, y - padding, x + w + padding, y + h + padding]
    else:
        bg_box = [x, y, x + w + padding * 2, y + h + padding * 2]
        x += padding
        y += padding

    # 반투명 배경 (Alpha = 180/255)
    r, g, b = bg_color
    draw.rectangle(bg_box, fill=(r, g, b, 180))
    
    # 텍스트
    draw.text((x, y), text, font=font, fill=(*text_color, 255))
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    bg_h, bg_w, _ = background.shape
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size)
    h, w, c = overlay.shape
    if x >= bg_w or y >= bg_h: return background
    h = min(h, bg_h - y)
    w = min(w, bg_w - x)
    if h <= 0 or w <= 0: return background
    overlay = overlay[:h, :w]
    if c == 4:
        alpha = overlay[:, :, 3] / 255.0
        for i in range(3): 
            background[y:y+h, x:x+w, i] = (1. - alpha) * background[y:y+h, x:x+w, i] + \
                                          alpha * overlay[:, :, i]
    else:
        background[y:y+h, x:x+w] = overlay
    return background

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
# 4. YOLO 분석 로직 (V8 유지)
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
    print("--- MERMAID AI COACH (V9: UI Design Upgrade) ---")
    
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

    speak_message("머메이드 코칭을 시작합니다. 왼쪽 사진을 보고 자세를 따라해 주세요.")

    frame_num = 0
    is_recording = False
    
    current_speech_state = "INIT" 
    new_speech_state = "INIT"
    
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
            
            # UI용 변수 초기화
            badge_text = "대기 중"
            badge_color = (100, 100, 100) # 회색
            subtitle_text = "왼쪽 모델 자세를 따라해 보세요"
            border_color = None

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                frame = results[0].plot(img=frame)
                kps = results[0].keypoints.data[0].cpu().numpy()

                if check_basic_pose(kps):
                    is_sitting = check_floor_sitting(kps)
                    if is_sitting:
                        spine_angle, es_dist, torso_len, curr_pose_coords, shoulder_level = calculate_mermaid_metrics(kps)
                        
                        pose_history.append(curr_pose_coords)
                        if len(pose_history) == 5:
                            smooth_pose = np.mean(pose_history, axis=0)
                            movement_score = calculate_movement(prev_smooth_pose, smooth_pose, torso_len)
                            prev_smooth_pose = smooth_pose 
                            if movement_score < MOVEMENT_THRESH: stable_frame_count += 1
                            else: stable_frame_count = 0 
                        else: movement_score = 100.0

                        if spine_angle < START_BENDING_THRESH:
                            # [READY]
                            if is_stabilized:
                                badge_text = "준비 완료"
                                badge_color = (0, 180, 0) # 녹색
                                subtitle_text = "옆으로 천천히 숙이세요"
                                log_entry['State'] = 'Ready_Locked'
                                new_speech_state = "READY_COMPLETE"
                            else:
                                is_strict_ready = (spine_angle < STRICT_READY_ANGLE) and (shoulder_level < SHOULDER_LEVEL_THRESH)
                                if not is_strict_ready:
                                    badge_text = "자세 교정 필요"
                                    badge_color = (0, 165, 255) # 주황
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
                                    # 게이지 바 디자인 (하단에 얇고 깔끔하게)
                                    bar_w = int(w * 0.6 * progress)
                                    cv2.rectangle(frame, (int(w*0.2), h-25), (int(w*0.2)+bar_w, h-15), (0, 255, 0), -1)
                                    cv2.rectangle(frame, (int(w*0.2), h-25), (int(w*0.2)+int(w*0.6), h-15), (200, 200, 200), 1)

                        else:
                            # [EXERCISE]
                            is_stabilized = True 
                            new_speech_state = "EXERCISE"
                            log_entry['State'] = 'Exercise'
                            sensor_data = get_mpu_data()
                            pelvis_status, pelvis_diff = check_pelvis_sensor(sensor_data)
                            
                            warning_list = []
                            if pelvis_status == "LEFT_UP": warning_list.append((f"왼쪽 골반 들림", "왼쪽 골반을 내려주세요"))
                            elif pelvis_status == "RIGHT_UP": warning_list.append((f"오른쪽 골반 들림", "오른쪽 골반을 내려주세요"))
                            if es_dist < EAR_SHOULDER_THRESH: warning_list.append((f"어깨 으쓱 주의", "어깨를 내려주세요"))
                            if spine_angle > SPINE_ANGLE_LIMIT: warning_list.append((f"과도한 꺾기", "너무 많이 꺾지 마세요"))
                            
                            if not warning_list:
                                badge_text = f"운동 중 ({int(spine_angle)}°)"
                                badge_color = (0, 180, 0) # 녹색
                                subtitle_text = "자세가 아주 좋습니다!"
                            else:
                                display_text, voice_text = warning_list[0]
                                badge_text = "자세 주의"
                                badge_color = (200, 0, 0) # 빨강
                                subtitle_text = display_text
                                border_color = (0, 0, 255)
                                trigger_voice_feedback(voice_text)

                            log_entry.update({'Spine_Angle': spine_angle, 'Warning': warning_list[0][0] if warning_list else "None"})
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

                # === [디자인 적용] 텍스트 그리기 ===
                # 1. 상태 배지 (좌측 상단)
                frame = draw_ui_text(frame, badge_text, (20, 20), 24, bg_color=badge_color, align="left")
                
                # 2. 피드백 자막 (중앙 하단)
                # 자막 배경은 항상 검정 반투명
                frame = draw_ui_text(frame, subtitle_text, (w//2, h-60), 32, bg_color=(0,0,0), align="center")
                
                if border_color:
                    cv2.rectangle(frame, (0,0), (w, h), border_color, 15)
            
            # 음성 상태 관리
            if new_speech_state != "INIT" and new_speech_state != current_speech_state:
                if new_speech_state == "STANDBY": speak_message("전신이 보이게 앉아주세요.")
                elif new_speech_state == "STABILIZING": speak_message("확인되었습니다. 움직이지 마세요.")
                elif new_speech_state == "READY_COMPLETE": speak_message("준비 완료. 머메이드 동작을 시작하세요.")
                current_speech_state = new_speech_state
            
            # 2분할 화면 병합
            final_display = frame
            if guide_img_resized is not None:
                final_display = np.hstack((guide_img_resized, frame))
                # 가이드 쪽 텍스트 (이미지가 있어서 흰색 글씨 잘 보임)
                cv2.putText(final_display, "GUIDE POSE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            output_w = 1280
            aspect_ratio = final_display.shape[0] / final_display.shape[1]
            output_h = int(output_w * aspect_ratio)
            final_display_resized = cv2.resize(final_display, (output_w, output_h))

            cv2.imshow(WINDOW_NAME, final_display_resized)
            
            if is_recording and video_recorder:
                if video_recorder.write_frame(frame, current_timestamp): 
                    if len(log_entry) > 2: recorded_data_log.append(log_entry)

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

#디자인 대충 픽스 - 글자 크기 줄이고, 자막 여백 위에 조금 있는데 수정해보고, 대사 조금 수정