import cv2
import numpy as np
from ultralytics import YOLO
import joblib
from collections import deque
import pyttsx3
import threading
import queue
import time
import os
import requests
import math
import pandas as pd
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

# 윈도우 소리 설정
try:
    import pythoncom
except ImportError:
    pass

# ===============================
# 1. 설정값
# ===============================
GUIDE_IMAGE_PATH = 'lunge/lunge.png'
SEQUENCE_LENGTH = 30

ENTER_THRESHOLD = 130  
EXIT_THRESHOLD = 150   
RESET_THRESHOLD = 165  

# 목표 각도 (이 각도에 가까울수록 점수 높음)
TARGET_ANGLE = 100.0 

HOLD_DURATION = 2.0
TRANSITION_DURATION = 4.0
PASS_SCORE = 40.0

SENSOR_URL = "http://192.168.4.1"
SENSOR_TIMEOUT = 0.05
PELVIS_THRESH = 5.0 # 경고 기준 (화면 표시용)

# ===============================
# 2. 디자인 유틸리티 (폰트 설정)
# ===============================
def get_font(size):
    try: return ImageFont.truetype("malgun.ttf", size)
    except: return ImageFont.load_default()

# ===============================
# 3. TTS 시스템
# ===============================
tts_queue = queue.Queue()
IS_SPEAKING = False 

def tts_worker_thread():
    global IS_SPEAKING
    while True:
        msg = tts_queue.get()
        if msg is None: break 
        
        IS_SPEAKING = True 
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except: pass

            print(f"🔊 [말함] {msg}")
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(msg)
            engine.runAndWait()
            engine.stop()
            del engine
            time.sleep(0.3)
        except Exception as e: 
            print(f"TTS 오류: {e}")
        finally:
            IS_SPEAKING = False 
            tts_queue.task_done()

threading.Thread(target=tts_worker_thread, daemon=True).start()

def speak(text):
    global IS_SPEAKING
    IS_SPEAKING = True 
    tts_queue.put(text)

# ===============================
# 4. 유틸리티 & 녹화
# ===============================
class FPSCalibrator:
    def __init__(self, calibration_frames=60):
        self.calibration_frames = calibration_frames
        self.frame_times = deque(maxlen=calibration_frames)
        self.prev_time = time.perf_counter()
    def update(self):
        current_time = time.perf_counter()
        delta = current_time - self.prev_time
        if delta > 0.001: self.frame_times.append(delta)
        self.prev_time = current_time
    def get_best_fps(self):
        if len(self.frame_times) > 5: return 1.0 / np.mean(list(self.frame_times))
        return 30.0

class VideoRecorder:
    def __init__(self, width, height, fps):
        self.writer = None; self.width = width; self.height = height; self.fps = fps; self.is_recording = False
        self.frame_interval = 1.0 / fps; self.last_write_time = None
    def start_recording(self):
        filename = f"Lunge_Video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        print(f"[REC] 시작: {filename} (FPS: {self.fps:.2f})")
        self.writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        self.is_recording = True; self.last_write_time = time.perf_counter()
    def write_frame(self, frame):
        if self.writer and self.is_recording:
            current_time = time.perf_counter()
            if self.last_write_time is None or (current_time - self.last_write_time) >= self.frame_interval:
                self.writer.write(frame); self.last_write_time = current_time
    def stop_recording(self):
        if self.writer: self.writer.release(); self.is_recording = False; print("[REC] 저장 완료")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

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
    diff = get_roll(m1) - get_roll(m2)
    status = "WARNING" if abs(diff) > PELVIS_THRESH else "LEVEL"
    return status, abs(diff)

# ===============================
# 5. 런지 코치 (규칙 기반)
# ===============================
class LungeCoach:
    def __init__(self):
        print("시스템 로딩 중...")
        self.yolo = YOLO('yolo11s-pose.pt')
        
        self.state = "INIT"
        self.state_start_time = time.time()
        self.hold_start_time = None
        self.fail_start_time = None
        self.transition_start_time = None
        self.setup_stable_start = None 

        self.current_set = 1
        self.total_sets = 3
        self.score_history = [] 
        self.angle_buffer = []
        
        self.smooth_l = None
        self.smooth_r = None
        self.last_score = 0.0
        self.sensor_status = "LEVEL"
        self.sensor_diff = 0.0

        self.guide_img_raw = None
        if os.path.exists(GUIDE_IMAGE_PATH):
            self.guide_img_raw = cv2.imread(GUIDE_IMAGE_PATH)

    def calculate_score_rule(self):
        if not self.angle_buffer: return 0.0
        
        # 1. 기본 점수 (무릎 각도)
        avg_angle = np.mean(self.angle_buffer)
        diff_angle = abs(avg_angle - TARGET_ANGLE)
        angle_score = 100 - (diff_angle * 1.5)
        
        # 2. 센서 감점 (5도당 3점)
        pelvis_penalty = (self.sensor_diff // 5.0) * 3.0
        
        if pelvis_penalty > 0:
            print(f"⚠️ 골반 감점: -{pelvis_penalty:.1f} (기울기: {self.sensor_diff:.1f}도)")

        final_score = angle_score - pelvis_penalty
        return max(0, min(100, final_score))

    def update_smooth_angle(self, new_val, old_val):
        if old_val is None: return new_val
        return 0.7 * old_val + 0.3 * new_val

    def draw_report(self, canvas, cam_w):
        x_start = canvas.shape[1] - cam_w
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x_start, 0), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        avg_score = sum(self.score_history) / len(self.score_history) if self.score_history else 0
        cx = x_start + 50
        cv2.putText(canvas, "WORKOUT COMPLETE", (cx, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"Avg Score: {avg_score:.1f}", (cx, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        y_pos = 200
        for i, score in enumerate(self.score_history):
            if i < 6:
                text = f"Set {(i//2)+1} {'L' if i%2==0 else 'R'}: {score:.1f}"
                cv2.putText(canvas, text, (cx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                y_pos += 40
        return canvas

    def process_frame(self, frame, frame_count):
        h, w = frame.shape[:2]
        
        log_data = {'Frame': frame_count, 'Timestamp': time.time(), 'State': self.state, 
                    'L_Knee': 0, 'R_Knee': 0, 'Pelvis_Diff': 0.0, 'Score': self.last_score}

        if self.guide_img_raw is not None:
            gh, gw = self.guide_img_raw.shape[:2]
            new_gw = int(gw * (h / gh))
            guide_resized = cv2.resize(self.guide_img_raw, (new_gw, h))
        else:
            guide_resized = np.zeros((h, 300, 3), dtype=np.uint8)
            new_gw = 300

        sensor_json = get_mpu_data()
        self.sensor_status, self.sensor_diff = check_pelvis_sensor(sensor_json)
        log_data['Pelvis_Diff'] = self.sensor_diff

        results = self.yolo(frame, verbose=False)
        current_time = time.time()
        detected = False
        
        if results[0].keypoints and len(results[0].keypoints) > 0:
            kpts = results[0].keypoints.xyn[0].cpu().numpy()
            
            # 좌우 반전 (거울 모드)
            l_hip, r_hip = kpts[12], kpts[11]
            l_knee, r_knee = kpts[14], kpts[13]
            l_ankle, r_ankle = kpts[16], kpts[15]

            if l_knee[0] > 0 and r_knee[0] > 0 and l_ankle[0] > 0 and r_ankle[0] > 0:
                detected = True
                
                raw_l = calculate_angle(l_hip, l_knee, l_ankle)
                raw_r = calculate_angle(r_hip, r_knee, r_ankle)
                
                self.smooth_l = self.update_smooth_angle(raw_l, self.smooth_l)
                self.smooth_r = self.update_smooth_angle(raw_r, self.smooth_r)
                
                log_data['L_Knee'] = self.smooth_l
                log_data['R_Knee'] = self.smooth_r

                cv2.putText(frame, f"L:{int(self.smooth_l)}", (int(l_knee[0]*w), int(l_knee[1]*h)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(frame, f"R:{int(self.smooth_r)}", (int(r_knee[0]*w), int(r_knee[1]*h)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                
                for kp in [l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]:
                    x, y = int(kp[0]*w), int(kp[1]*h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        canvas = np.hstack((guide_resized, frame))
        
        # [기존 상단 블랙바]
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)

        # -----------------------------------------------------------------
        # [수정된 부분] 텍스트 그리기 로직 (PIL) - 진한 갈색, 상단 중앙, Bold
        # -----------------------------------------------------------------
        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_guide = get_font(30)

        # 1. 진한 갈색 색상 정의 (RGB)
        dark_brown = (101, 67, 33)

        # 2. 텍스트 크기 계산
        text = "자세를 취해주세요"
        bbox = draw.textbbox((0, 0), text, font=font_guide)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 3. 위치 계산 (왼쪽 이미지 영역의 상단 중앙, y=50)
        # guide_resized 너비는 new_gw
        x_pos = (new_gw - text_w) // 2
        y_pos = 50 

        # 4. 텍스트 그리기 (진한 갈색, Bold 처리)
        draw.text((x_pos, y_pos), text, font=font_guide, fill=dark_brown, stroke_width=1, stroke_fill=dark_brown)
        
        canvas = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # -----------------------------------------------------------------

        global IS_SPEAKING
        if detected:
            if IS_SPEAKING:
                 if self.state in ["LEFT_HOLD", "RIGHT_HOLD"] and self.hold_start_time:
                      elapsed = current_time - self.hold_start_time
                      bar_width = int((elapsed / HOLD_DURATION) * 200)
                      cv2.rectangle(canvas, (new_gw + 50, 100), (new_gw + 50 + bar_width, 130), (0, 255, 255), -1)
                      cv2.rectangle(canvas, (new_gw + 50, 100), (new_gw + 250, 130), (255, 255, 255), 2)
            else:
                if self.state == "INIT":
                    if current_time - self.state_start_time > 1.0:
                        speak("런지 자세 평가를 시작합니다. 전신이 보이게 뒤로 서주세요.")
                        self.state = "BODY_CHECK"

                elif self.state == "BODY_CHECK":
                    is_standing = self.smooth_l > RESET_THRESHOLD and self.smooth_r > RESET_THRESHOLD
                    is_visible = l_ankle[1] < 0.95 and r_ankle[1] < 0.95
                    if is_standing and is_visible:
                        if self.setup_stable_start is None: self.setup_stable_start = current_time
                        elif current_time - self.setup_stable_start > 1.5:
                            speak("확인되었습니다. 이제 측면으로 서주세요.")
                            self.state = "SIDE_CHECK"
                            self.setup_stable_start = None
                    else:
                        self.setup_stable_start = None
                        if not is_visible: cv2.putText(frame, "MOVE BACK", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        elif not is_standing: cv2.putText(frame, "STAND UP", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                elif self.state == "SIDE_CHECK":
                    if abs(l_ankle[0] - r_ankle[0]) > 0.05:
                        speak("평가를 시작합니다. 왼다리를 앞으로 움직여주세요.")
                        self.state = "LEFT_READY"
                        self.angle_buffer = []
                    else:
                        cv2.putText(frame, "TURN SIDEWAYS", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # --- 왼쪽 ---
                elif self.state == "LEFT_READY":
                    if self.smooth_l < ENTER_THRESHOLD:
                        self.state = "LEFT_HOLD"
                        self.hold_start_time = current_time
                        self.fail_start_time = None
                        self.angle_buffer = []
                        speak("버티세요!")

                elif self.state == "LEFT_HOLD":
                    elapsed = current_time - self.hold_start_time
                    self.angle_buffer.append(self.smooth_l)
                    
                    if self.smooth_l > EXIT_THRESHOLD:
                        if self.fail_start_time is None: self.fail_start_time = current_time
                        elif current_time - self.fail_start_time > 0.5:
                            self.state = "LEFT_READY"
                            speak("자세가 풀렸습니다.")
                    else:
                        self.fail_start_time = None

                    bar_width = int((elapsed / HOLD_DURATION) * 200)
                    cv2.rectangle(canvas, (new_gw + 50, 100), (new_gw + 50 + bar_width, 130), (0, 255, 255), -1)
                    cv2.rectangle(canvas, (new_gw + 50, 100), (new_gw + 250, 130), (255, 255, 255), 2)

                    if elapsed >= HOLD_DURATION:
                        score = self.calculate_score_rule()
                        self.last_score = score
                        
                        if score >= PASS_SCORE:
                            self.score_history.append(score)
                            speak("좋습니다. 오른 다리 준비하세요.")
                            self.state = "SWITCH_TO_RIGHT"
                            self.transition_start_time = current_time
                        else:
                            speak("자세가 너무 높습니다. 더 앉으세요.")
                            self.state = "RETRY_WAIT_L"
                            self.transition_start_time = current_time
                        self.angle_buffer = []

                elif self.state == "SWITCH_TO_RIGHT":
                    remaining = TRANSITION_DURATION - (current_time - self.transition_start_time)
                    if remaining <= 0 and self.smooth_r > RESET_THRESHOLD:
                        self.state = "RIGHT_READY"
                        speak("오른 다리 시작.")

                elif self.state == "RETRY_WAIT_L":
                    remaining = TRANSITION_DURATION - (current_time - self.transition_start_time)
                    if remaining <= 0 and self.smooth_l > RESET_THRESHOLD:
                        self.state = "LEFT_READY"
                        speak("왼다리 시작.")

                # --- 오른쪽 ---
                elif self.state == "RIGHT_READY":
                    if self.smooth_r < ENTER_THRESHOLD:
                        self.state = "RIGHT_HOLD"
                        self.hold_start_time = current_time
                        self.fail_start_time = None
                        self.angle_buffer = []
                        speak("버티세요!")

                elif self.state == "RIGHT_HOLD":
                    elapsed = current_time - self.hold_start_time
                    self.angle_buffer.append(self.smooth_r)
                    
                    if self.smooth_r > EXIT_THRESHOLD:
                        if self.fail_start_time is None: self.fail_start_time = current_time
                        elif current_time - self.fail_start_time > 0.5:
                            self.state = "RIGHT_READY"
                            speak("자세가 풀렸습니다.")
                    else:
                        self.fail_start_time = None

                    bar_width = int((elapsed / HOLD_DURATION) * 200)
                    cv2.rectangle(canvas, (new_gw + 50, 100), (new_gw + 50 + bar_width, 130), (0, 255, 255), -1)
                    cv2.rectangle(canvas, (new_gw + 50, 100), (new_gw + 250, 130), (255, 255, 255), 2)

                    if elapsed >= HOLD_DURATION:
                        score = self.calculate_score_rule()
                        self.last_score = score
                        
                        if score >= PASS_SCORE:
                            self.score_history.append(score)
                            if self.current_set < self.total_sets:
                                speak(f"정상입니다. 다음 세트 준비.")
                                self.state = "SWITCH_TO_NEXT_SET"
                            else:
                                speak("모든 평가가 완료되었습니다.")
                                self.state = "END"
                            self.transition_start_time = current_time
                        else:
                            speak("자세가 높습니다. 다시 하세요.")
                            self.state = "RETRY_WAIT_R"
                            self.transition_start_time = current_time
                        self.angle_buffer = []

                elif self.state == "SWITCH_TO_NEXT_SET":
                    remaining = TRANSITION_DURATION - (current_time - self.transition_start_time)
                    if remaining <= 0 and self.smooth_l > RESET_THRESHOLD:
                        self.current_set += 1
                        self.state = "LEFT_READY"
                        speak("왼다리 준비.")

                elif self.state == "RETRY_WAIT_R":
                    remaining = TRANSITION_DURATION - (current_time - self.transition_start_time)
                    if remaining <= 0 and self.smooth_r > RESET_THRESHOLD:
                        self.state = "RIGHT_READY"
                        speak("오른다리 시작.")

        if not detected:
            cv2.putText(frame, "BODY NOT DETECTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        info_text = f"Set {self.current_set}/{self.total_sets} | {self.state}"
        if IS_SPEAKING: info_text += " [Speaking...]"
        cv2.putText(canvas, info_text, (new_gw + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.sensor_status == "WARNING":
            cv2.putText(canvas, f"PELVIS WARNING ({self.sensor_diff:.1f})", (new_gw + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            log_data['Warning'] = 'Pelvis Tilt'
        
        score_text = f"Last Score: {self.last_score:.1f}"
        score_color = (0, 255, 0) if self.last_score >= PASS_SCORE else (0, 0, 255)
        cv2.putText(canvas, score_text, (new_gw + 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

        if self.state == "END":
            canvas = self.draw_report(canvas, w)
            
        return canvas, log_data

def main():
    coach = LungeCoach()
    cap = cv2.VideoCapture(0)
    calibrator = FPSCalibrator()
    recorder = None
    all_logs = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        calibrator.update()
        frame_count += 1
        frame = cv2.flip(frame, 1)
        
        final_view, log_data = coach.process_frame(frame, frame_count)
        
        if recorder and recorder.is_recording:
            recorder.write_frame(final_view)
            all_logs.append(log_data)
            cv2.circle(final_view, (final_view.shape[1] - 30, 30), 10, (0, 0, 255), -1)

        cv2.imshow('AI Lunge Coach - Final V17', final_view)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('r'):
            if recorder is None:
                real_fps = calibrator.get_best_fps()
                h, w = final_view.shape[:2]
                recorder = VideoRecorder(w, h, real_fps)
            if recorder.is_recording:
                recorder.stop_recording()
                if all_logs:
                    csv_name = f"Lunge_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    pd.DataFrame(all_logs).to_csv(csv_name, index=False)
                    print(f"[LOG] 저장 완료: {csv_name}")
                    all_logs = []
            else:
                recorder.start_recording()

    cap.release()
    if recorder and recorder.is_recording: recorder.stop_recording()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()