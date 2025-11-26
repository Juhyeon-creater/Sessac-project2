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
EAR_SHOULDER_THRESH = 0.15  # [수정] 민감도 완화 (0.28 -> 0.15)
SPINE_ANGLE_LIMIT = 45
START_BENDING_THRESH = 15   
CENTER_ANGLE_THRESH = 10.0  
SHOULDER_LEVEL_THRESH = 0.1 
PELVIS_SENSOR_THRESH = 15.0 # [수정] 민감도 완화 (5.0 -> 15.0)

# [운동 루틴 설정]
TARGET_SETS = 3
HOLD_DURATION = 5.0 

SENSOR_URL = "http://192.168.4.1"
SENSOR_TIMEOUT = 0.05
WINDOW_NAME = "Mermaid AI Coach (Routine Mode)"
CALIBRATION_FRAMES = 60

GUIDE_IMAGE_PATH = "mermaid/reference_pose.png" 

# ===============================
# 2. 디자인 유틸리티 (UI/UX)
# ===============================
def get_font(size):
    try: return ImageFont.truetype("malgun.ttf", size) 
    except: return ImageFont.load_default()

def draw_ui_text(img, text, pos, font_size, bg_color=(0,0,0), text_color=(255,255,255), align="left"):
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
    if tts_queue.qsize() < 3: tts_queue.put(message)

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
        if self.is_calibrated: return self.measured_fps
        elif len(self.frame_times) > 5: return 1.0 / np.mean(list(self.frame_times))
        else: return 30.0

class VideoRecorder:
    def __init__(self, width, height, fps):
        self.writer = None; self.fps = fps; self.width = width; self.height = height
        self.is_recording = False; self.last_write_time = None; self.frame_interval = 1.0 / fps
    def start_recording(self):
        filename = f"Mermaid_Video_{datetime.now().strftime('%H%M%S')}.mp4"
        print(f"[녹화 시작] {filename}")
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
        if self.writer: self.writer.release(); self.is_recording = False

# ===============================
# 4. 머메이드 로직 클래스 (상태 머신)
# ===============================
class MermaidRoutine:
    def __init__(self):
        self.state = "IDLE" 
        self.current_set = 1
        self.target_sets = TARGET_SETS
        self.hold_duration = HOLD_DURATION
        
        self.state_start_time = 0
        self.last_feedback_time = 0
        self.has_started = False

    # [추가] 숫자 -> 한글 발음 변환
    def get_kor_set(self):
        mapping = {1: "일", 2: "이", 3: "삼", 4: "사", 5: "오"}
        return mapping.get(self.current_set, str(self.current_set))

    def update(self, angle, direction, is_stable, now):
        # 리턴: (Badge_Text, Badge_Color, Subtitle_Text, Voice_Cmd, Border_Color)
        
        # 1. [IDLE] 초기 안정화 대기
        if self.state == "IDLE":
            if is_stable:
                if not self.has_started:
                    self.has_started = True
                    self.state = "PREP"
                    self.state_start_time = now
                    return "준비 완료", (0, 180, 0), "AI 코칭을 시작합니다", "AI 코칭을 시작합니다.", None
                else:
                    self.state = "LEFT_WAIT"
                    self.state_start_time = now
                    return "준비 완료", (0, 180, 0), "왼쪽부터 시작합니다", "왼쪽으로 몸을 기울여 주세요.", None
            else:
                return "안정화 중...", (0, 165, 255), "움직이지 마세요", None, None

        # 2. [PREP] 시작 멘트 대기
        if self.state == "PREP":
            if now - self.state_start_time > 3.0:
                self.state = "LEFT_WAIT"
                # [수정] 일세트 시작
                voice_msg = f"{self.get_kor_set()}세트 시작. 왼쪽으로 몸을 기울여 주세요."
                return "1세트 시작", (0, 255, 255), "왼쪽으로 기울이세요", voice_msg, None
            return "준비", (200, 200, 200), "잠시 후 시작합니다", None, None

        # 3. [LEFT_WAIT] 왼쪽 굽히기 대기
        if self.state == "LEFT_WAIT":
            if angle > START_BENDING_THRESH:
                if direction == "LEFT":
                    self.state = "LEFT_HOLD"
                    self.state_start_time = now
                    return "왼쪽 유지", (0, 255, 0), "5초간 유지하세요", "좋습니다. 유지하세요.", None
                else:
                    return "방향 주의", (0, 0, 255), "반대쪽(왼쪽)입니다!", "왼쪽입니다.", (0, 0, 255)
            return f"{self.current_set}세트 (왼쪽)", (0, 255, 255), "왼쪽으로 내려가세요", None, None

        # 4. [LEFT_HOLD] 왼쪽 5초 유지
        if self.state == "LEFT_HOLD":
            elapsed = now - self.state_start_time
            remain = self.hold_duration - elapsed
            
            if angle < START_BENDING_THRESH:
                self.state = "LEFT_WAIT" 
                return "자세 풀림", (0, 0, 255), "다시 내려가세요", "자세가 풀렸습니다.", (0, 0, 255)

            if remain <= 0:
                self.state = "RECOVER_1"
                return "완료", (0, 180, 0), "제자리로 돌아오세요", "다시 정자세로 돌아와 주세요.", None
            
            return f"왼쪽 버티기 {int(remain)+1}초", (0, 255, 0), "", None, None

        # 5. [RECOVER_1] 왼쪽 완료 후 복귀 대기
        if self.state == "RECOVER_1":
            if angle < CENTER_ANGLE_THRESH:
                self.state = "RIGHT_WAIT"
                time.sleep(1.0) 
                return "오른쪽 준비", (0, 255, 255), "이제 오른쪽입니다", "오른쪽으로 몸을 기울여 주세요.", None
            return "복귀 중", (200, 200, 200), "허리를 세우세요", None, None

        # 6. [RIGHT_WAIT] 오른쪽 굽히기 대기
        if self.state == "RIGHT_WAIT":
            if angle > START_BENDING_THRESH:
                if direction == "RIGHT":
                    self.state = "RIGHT_HOLD"
                    self.state_start_time = now
                    return "오른쪽 유지", (0, 255, 0), "5초간 유지하세요", "좋습니다. 유지하세요.", None
                else:
                    return "방향 주의", (0, 0, 255), "반대쪽(오른쪽)입니다!", "오른쪽입니다.", (0, 0, 255)
            return f"{self.current_set}세트 (오른쪽)", (0, 255, 255), "오른쪽으로 내려가세요", None, None

        # 7. [RIGHT_HOLD] 오른쪽 5초 유지
        if self.state == "RIGHT_HOLD":
            elapsed = now - self.state_start_time
            remain = self.hold_duration - elapsed
            
            if angle < START_BENDING_THRESH:
                self.state = "RIGHT_WAIT"
                return "자세 풀림", (0, 0, 255), "다시 내려가세요", "자세가 풀렸습니다.", (0, 0, 255)

            if remain <= 0:
                self.state = "RECOVER_2"
                return "완료", (0, 180, 0), "제자리로 돌아오세요", "다시 정자세로 돌아와 주세요.", None
            
            return f"오른쪽 버티기 {int(remain)+1}초", (0, 255, 0), "", None, None

        # 8. [RECOVER_2] 오른쪽 완료 후 복귀 (세트 끝)
        if self.state == "RECOVER_2":
            if angle < CENTER_ANGLE_THRESH:
                if self.current_set >= self.target_sets:
                    self.state = "FINISHED"
                    return "운동 종료", (0, 180, 0), "모든 세트 완료!", "수고하셨습니다. 모든 운동을 마쳤습니다.", None
                else:
                    self.current_set += 1
                    self.state = "LEFT_WAIT"
                    time.sleep(1.0)
                    # [수정] 다음 세트 발음 (일, 이, 삼)
                    voice_msg = f"{self.get_kor_set()}세트 시작. 왼쪽으로 기울여 주세요."
                    return f"{self.current_set}세트 시작", (0, 255, 255), "다음 세트 왼쪽 준비", voice_msg, None
            return "복귀 중", (200, 200, 200), "허리를 세우세요", None, None

        # 9. [FINISHED]
        if self.state == "FINISHED":
            return "종료", (100, 100, 100), "운동 완료", None, None

        return "대기", (100, 100, 100), "", None, None

# ===============================
# 5. YOLO 및 계산 로직
# ===============================
def calculate_mermaid_metrics(kps):
    # 키포인트 추출
    l_sh, r_sh = kps[5][:2], kps[6][:2]
    l_ear, r_ear = kps[3][:2], kps[4][:2]
    l_hip, r_hip = kps[11][:2], kps[12][:2]
    
    # 중심점
    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    
    # 몸통 길이 (정규화용)
    torso_len = np.linalg.norm(mid_sh - mid_hip)
    if torso_len == 0: return 0, 0, 0, np.zeros((4, 2)), 0, "CENTER"

    # 1. 척추 각도 (Spine Angle)
    spine_vec = mid_sh - mid_hip
    vertical_vec = np.array([0, -1])
    spine_u = spine_vec / np.linalg.norm(spine_vec)
    dot_prod = np.dot(spine_u, vertical_vec)
    spine_angle = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))

    # 2. 방향 판별 (Direction)
    direction = "CENTER"
    if spine_angle > START_BENDING_THRESH:
        if mid_sh[0] < mid_hip[0]: direction = "LEFT"
        else: direction = "RIGHT"

    # 3. 귀-어깨 거리
    l_dist = np.linalg.norm(l_ear - l_sh) / torso_len
    r_dist = np.linalg.norm(r_ear - r_sh) / torso_len
    min_ear_sh_dist = min(l_dist, r_dist)

    # 4. 어깨 수평
    shoulder_level = abs(l_sh[1] - r_sh[1]) / torso_len
    
    current_pose = np.array([l_sh, r_sh, l_hip, r_hip])
    
    return spine_angle, min_ear_sh_dist, torso_len, current_pose, shoulder_level, direction

def check_basic_pose(kps):
    nose_conf = kps[0][2]
    hip_conf = min(kps[11][2], kps[12][2])
    return nose_conf > 0.5 and hip_conf > 0.5

def check_floor_sitting(kps):
    l_hip_y, r_hip_y = kps[11][1], kps[12][1]
    l_knee_y, r_knee_y = kps[13][1], kps[14][1]
    hip_y_avg = (l_hip_y + r_hip_y) / 2
    knee_y_avg = (l_knee_y + r_knee_y) / 2
    return knee_y_avg > hip_y_avg - 50 # 여유값

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
# 6. 메인 실행 루프
# ===============================
def run_mermaid_coach():
    print("--- MERMAID AI COACH (Routine: 3Sets L/R) ---")
    
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

    speak_message("머메이드 코칭을 시작합니다. 전신이 보이게 앉아주세요.")

    frame_num = 0
    is_recording = False
    
    # 상태 관리 객체 생성
    mermaid = MermaidRoutine()
    
    pose_history = deque(maxlen=5) 
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

            results = model(frame, verbose=False, conf=0.5)
            log_entry = {'Timestamp': current_timestamp, 'Frame': frame_num}
            
            badge_text = "대기 중"
            badge_color = (100, 100, 100) 
            subtitle_text = "" 
            border_color = None

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                # [수정] boxes=False 옵션 추가 (파란 박스 제거)
                frame = results[0].plot(boxes=False, img=frame)
                kps = results[0].keypoints.data[0].cpu().numpy()

                if check_basic_pose(kps):
                    is_sitting = check_floor_sitting(kps)
                    if is_sitting:
                        # 1. 데이터 계산
                        spine_angle, es_dist, torso_len, curr_pose_coords, shoulder_level, direction = calculate_mermaid_metrics(kps)
                        sensor_data = get_mpu_data()
                        pelvis_status, pelvis_diff = check_pelvis_sensor(sensor_data)
                        
                        # 2. 안정화 체크
                        pose_history.append(curr_pose_coords)
                        if len(pose_history) == 5:
                            if stable_frame_count < REQUIRED_STABLE_FRAMES:
                                stable_frame_count += 1
                            else:
                                is_stabilized = True
                        
                        # 3. 상태 머신 업데이트
                        res_badge, res_col, res_sub, res_voice, res_border = mermaid.update(spine_angle, direction, is_stabilized, current_timestamp)
                        
                        badge_text = res_badge
                        badge_color = res_col
                        subtitle_text = res_sub
                        if res_border: border_color = res_border
                        if res_voice: speak_message(res_voice)

                        # 4. 운동 중일 때 경고 체크
                        if "HOLD" in mermaid.state:
                            warning_list = []
                            if pelvis_status == "LEFT_UP": warning_list.append(("Left Pelvis", "왼쪽 골반 누르세요"))
                            elif pelvis_status == "RIGHT_UP": warning_list.append(("Right Pelvis", "오른쪽 골반 누르세요"))
                            if es_dist < EAR_SHOULDER_THRESH: warning_list.append(("Shoulder", "어깨 내리세요"))
                            if spine_angle > SPINE_ANGLE_LIMIT: warning_list.append(("Too Much", "너무 많이 꺾지 마세요"))
                            
                            if warning_list:
                                w_text, w_voice = warning_list[0]
                                border_color = (0, 0, 255)
                                trigger_voice_feedback(w_voice)
                                log_entry['Warning'] = w_text

                        # 5. 로그 저장
                        log_entry.update({
                            'Spine_Angle': spine_angle, 
                            'Direction': direction,
                            'Pelvis_Tilt': pelvis_diff,
                            'State': mermaid.state,
                            'Set': mermaid.current_set,
                            'Warning': log_entry.get('Warning', 'None')
                        })
                        
                        if is_recording:
                            recorded_data_log.append(log_entry)

                    else:
                        badge_text = "대기 중"
                        subtitle_text = "바닥에 편하게 앉아주세요"
                        is_stabilized = False
                        stable_frame_count = 0
                else:
                    badge_text = "인식 불가"
                    subtitle_text = "전신이 보이게 앉아주세요"
                    is_stabilized = False
                    stable_frame_count = 0
                    pose_history.clear()

                # UI 그리기
                frame = draw_ui_text(frame, badge_text, (20, 20), 24, bg_color=badge_color, align="left")
                if subtitle_text:
                    frame = draw_ui_text(frame, subtitle_text, (w//2, h-50), 20, bg_color=(0,0,0), align="center")
                
                # [수정] 빨간 테두리 제거 (주석 처리)
                # if border_color:
                #    cv2.rectangle(frame, (0,0), (w, h), border_color, 15)
            
            # 화면 병합 및 출력
            final_display = frame
            if guide_img_resized is not None:
                final_display = np.hstack((guide_img_resized, frame))
            
            # -----------------------------------------------------------------
            # [수정된 부분] 텍스트 그리기 로직 (PIL) - 진한 갈색, 상단 중앙, Bold
            # -----------------------------------------------------------------
            img_pil = Image.fromarray(cv2.cvtColor(final_display, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font_guide = get_font(30)

            # 1. 진한 갈색 색상 정의 (RGB)
            dark_brown = (101, 67, 33) #폰트색
            text_bg_color = (255, 248, 220) # 배경색 (크림색 예시)

            # 2. 텍스트 크기 계산
            text = "자세를 취해주세요"
            bbox = draw.textbbox((0, 0), text, font=font_guide)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # 3. 위치 계산 (왼쪽 이미지 영역의 상단 중앙)
            if guide_img_resized is not None:
                # 왼쪽 가이드 이미지의 너비
                w_guide = guide_img_resized.shape[1]
                # 왼쪽 영역 안에서 중앙 정렬
                x_pos = (w_guide - text_w) // 2
            else:
                # 가이드 이미지가 없을 경우 전체 화면의 1/4 지점(왼쪽 편)
                h_final, w_final = final_display.shape[:2]
                x_pos = (w_final // 4) - (text_w // 2)

            # 상단 위치 (위에서 50px 아래로)
            y_pos = 50 

            # --- [추가] 배경 사각형 그리기 ---
            padding_x = 10
            padding_y = 5
            # 배경 사각형 좌표 계산
            bg_bbox = (
                x_pos + bbox[0] - padding_x,
                y_pos + bbox[1] - padding_y,
                x_pos + bbox[2] + padding_x,
                y_pos + bbox[3] + padding_y
            )
            # 배경 그리기
            draw.rectangle(bg_bbox, fill=text_bg_color)
            # 4. 텍스트 그리기 (진한 갈색, Bold 처리: stroke_width 추가)
            draw.text((x_pos, y_pos), text, font=font_guide, fill=dark_brown, stroke_width=2, stroke_fill=dark_brown)
            
            final_display = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            # -----------------------------------------------------------------

            output_w = 1280
            aspect_ratio = final_display.shape[0] / final_display.shape[1]
            output_h = int(output_w * aspect_ratio)
            final_display_resized = cv2.resize(final_display, (output_w, output_h))
            
            if is_recording:
                cv2.circle(final_display_resized, (output_w - 30, 30), 10, (0, 0, 255), -1)

            cv2.imshow(WINDOW_NAME, final_display_resized)
            
            if is_recording and video_recorder:
                video_recorder.write_frame(frame, current_timestamp)

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