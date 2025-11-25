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

TARGET_ANGLE_MIN = 18.0

TARGET_ANGLE_MAX = 53.0

START_LIFT_THRESH = 10.0

PELVIS_SENSOR_THRESH = 5.0



# 헌드레드 운동 설정

HOLD_DURATION = 10.0

REST_DURATION = 5.0

TARGET_SETS = 3



SENSOR_URL = "http://192.168.4.1"

SENSOR_TIMEOUT = 0.05

WINDOW_NAME = "Hundred AI Coach (Korean)"

CALIBRATION_FRAMES = 60



GUIDE_IMAGE_PATH = "hundred/reference_pose.png" # 이미지 경로 설정



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

    draw.rectangle(bg_box, fill=(r, g, b, 153)) # 투명도 60%

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

    if tts_queue.qsize() < 3:

        tts_queue.put(message)



last_speech_time = 0

speech_cooldown = 3.0

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

        filename = f"Hundred_Video_{datetime.now().strftime('%H%M%S')}.mp4"

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



# ============================================================

#       ★ 헌드레드 로직 클래스 ★

# ============================================================

class HundredCoach:

    def __init__(self):

        self.PREP_VOICE_DELAY = 3.0

        self.HOLD_DURATION = HOLD_DURATION

        self.REST_DURATION = REST_DURATION

        self.TARGET_SETS = TARGET_SETS

        self.MIN_ANGLE = TARGET_ANGLE_MIN

        self.MAX_ANGLE = TARGET_ANGLE_MAX



        self.state = "IDLE"

        self.current_set = 1

        self.state_start_time = None

        self.last_feedback_time = 0



    def start_preparation(self, now):

        if self.state == "IDLE":

            self.state = "PREP"

            self.state_start_time = now

            return "자세 확인. 잠시 후 시작합니다."

        return None



    def reset(self):

        self.state = "IDLE"

        self.state_start_time = None



    def update(self, angle, now):

        # 리턴값: (Badge_Text, Badge_Color, Subtitle_Text, Voice_Command, Border_Color)

       

        # 1. [종료 상태]

        if self.state == "FINISHED":

            return "운동 완료", (0, 0, 255), "수고하셨습니다!", None, None



        # 2. [준비 상태]

        if self.state == "PREP":

            elapsed = now - self.state_start_time

            if elapsed < self.PREP_VOICE_DELAY:

                return "준비 중", (255, 255, 0), "잠시 후 시작합니다...", None, None

           

            self.state = "WAIT_LIFT"

            self.state_start_time = now

            return "시작", (0, 255, 255), "다리를 들어올리세요", "시작. 다리를 올리세요", None



        # 3. [다리 들기 대기]

        if self.state == "WAIT_LIFT":

            if angle >= self.MIN_ANGLE:

                self.state = "HOLD"

                self.state_start_time = now

                self.last_feedback_time = now

                return f"{self.current_set}세트 시작", (0, 255, 0), "버티기 시작!", f"{self.current_set}세트 시작", None

            else:

                return f"{self.current_set}세트 준비", (0, 165, 255), "다리를 목표 각도로 올리세요", None, None



        # 4. [휴식 상태]

        if self.state == "REST":

            elapsed = now - self.state_start_time

            remain = self.REST_DURATION - elapsed

            if remain <= 0:

                self.state = "WAIT_LIFT"

                self.current_set += 1

                self.state_start_time = now

                return f"{self.current_set}세트 준비", (0, 255, 255), "다시 다리를 올리세요", f"{self.current_set}세트 시작.", None

           

            return "휴식 중", (200, 200, 200), f"다음 세트까지 {int(remain)+8}초", None, None



        # 5. [운동(HOLD) 상태]

        if self.state == "HOLD":

            elapsed = now - self.state_start_time

           

            # 각도 피드백

            badge_txt = f"{self.current_set}세트 운동 중"

            badge_col = (0, 255, 0)

            sub_txt = "자세가 아주 좋습니다!"

            voice = None

            border = None



            if angle < self.MIN_ANGLE:

                badge_txt = "자세 주의 (Low)"

                badge_col = (0, 165, 255)

                sub_txt = "다리를 더 올리세요"

                border = (0, 165, 255)

                if now - self.last_feedback_time > 3.0:

                    voice = "다리를 더 올리세요"

                    self.last_feedback_time = now



            elif angle > self.MAX_ANGLE:

                badge_txt = "자세 주의 (High)"

                badge_col = (0, 165, 255)

                sub_txt = "다리를 조금 내리세요"

                border = (0, 165, 255)

                if now - self.last_feedback_time > 3.0:

                    voice = "다리를 조금만 내리세요"

                    self.last_feedback_time = now

           

            # 시간 체크

            if elapsed >= self.HOLD_DURATION:

                if self.current_set >= self.TARGET_SETS:

                    self.state = "FINISHED"

                    return "운동 완료", (255, 0, 0), "모든 세트 종료", "운동 끝. 수고하셨습니다.", None

                else:

                    self.state = "REST"

                    self.state_start_time = now

                    return "휴식", (200, 200, 200), "잠시 휴식하세요", "휴식.", None



            return badge_txt, badge_col, sub_txt, voice, border



        return "대기 중", (100, 100, 100), "...", None, None



# ===============================

# 6. YOLO 및 센서 로직

# ===============================

def calculate_leg_angle(kps):

    l_conf, r_conf = kps[15][2], kps[16][2]

    if l_conf >= r_conf: hip, ankle = kps[11][:2], kps[15][:2]

    else: hip, ankle = kps[12][:2], kps[16][:2]

    dy = -(ankle[1] - hip[1])

    dx = abs(ankle[0] - hip[0])

    return np.degrees(np.arctan2(dy, dx))



def check_lying_ready(kps):

    # 어깨와 골반이 모두 보여야 함

    valid_kps = (kps[5][2]>0.5 and kps[11][2]>0.5)

    if not valid_kps: return False

    l_sh, r_sh = kps[5][:2], kps[6][:2]

    l_hip, r_hip = kps[11][:2], kps[12][:2]

    mid_sh, mid_hip = (l_sh + r_sh) / 2, (l_hip + r_hip) / 2

    # 가로(x) 거리가 세로(y) 거리보다 커야 누운 것으로 간주

    dx = abs(mid_sh[0] - mid_hip[0])

    dy = abs(mid_sh[1] - mid_hip[1])

    return dx > dy



def get_mpu_data():

    try:

        r = requests.get(SENSOR_URL, timeout=SENSOR_TIMEOUT)

        if r.status_code == 200: return r.json()

    except: pass

    return None



def check_pelvis_sensor(sensor_data):

    if not sensor_data: return None, 0.0

    m1, m2 = sensor_data.get("mpu1"), sensor_data.get("mpu2")

    if not (m1 and m2): return None, 0.0

    def get_roll(m): return math.degrees(math.atan2(m.get("AcY", 0), m.get("AcZ", 1)))

    diff = abs(get_roll(m1) - get_roll(m2))

    return ("UNSTABLE" if diff > PELVIS_SENSOR_THRESH else "LEVEL"), diff



# ===============================

# 7. 메인 실행 루프

# ===============================

def run_hundred_coach():

    print("--- HUNDRED AI COACH (Design Ver) ---")

   

    guide_img_original = None

    if os.path.exists(GUIDE_IMAGE_PATH):

        guide_img_original = cv2.imread(GUIDE_IMAGE_PATH)

        if guide_img_original is None: print("[!] 가이드 이미지 로드 실패")

        else: print(f"[✓] 가이드 이미지 로드 성공")



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

   

    # 가이드 이미지 리사이징

    guide_img_resized = None

    if guide_img_original is not None:

        h_guide, w_guide = guide_img_original.shape[:2]

        scale = frame_height / h_guide

        guide_img_resized = cv2.resize(guide_img_original, (int(w_guide * scale), frame_height))



    speak_message("헌드레드 코칭을 시작합니다. 몸 옆면이 보이게 누워주세요.")



    frame_num = 0

    is_recording = False

    hundred = HundredCoach()

    lying_stable_start = None

   

    has_said_start = False



    try:

        while True:

            ret, frame = cap.read()

            if not ret: break

           

            # [수정됨] 좌우 반전 (거울 모드)

            frame = cv2.flip(frame, 1)



            frame_num += 1

            current_timestamp = time.time()

            h, w, _ = frame.shape

            calibrator.update()



            results = model(frame, verbose=False, conf=0.5)

            log_entry = {'Timestamp': current_timestamp, 'Frame': frame_num}

           

            # UI 변수 초기화

            badge_text = "대기 중"

            badge_color = (100, 100, 100)

            subtitle_text = "전신이 보이게 누워주세요"

            border_color = None



            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:

                frame = results[0].plot(img=frame)

                kps = results[0].keypoints.data[0].cpu().numpy()



                if check_lying_ready(kps):

                    angle = calculate_leg_angle(kps)

                    sensor_data = get_mpu_data()

                    pelvis_status, pelvis_diff = check_pelvis_sensor(sensor_data)

                   

                    log_entry.update({

                        'Leg_Angle': angle,

                        'Pelvis_Tilt': pelvis_diff,

                        'State': hundred.state,

                        'Set': hundred.current_set

                    })



                    # 상태 머신 진행

                    if hundred.state == "IDLE":

                        if lying_stable_start is None:

                            lying_stable_start = current_timestamp

                       

                        stable_time = current_timestamp - lying_stable_start

                        if stable_time < 1.5:

                            badge_text = "안정화 중..."

                            badge_color = (0, 165, 255)

                            subtitle_text = f"움직이지 마세요 {1.5 - stable_time:.1f}"

                        else:

                            if not has_said_start:

                                speak_message("자세 확인. 시작합니다.")

                                has_said_start = True

                            msg = hundred.start_preparation(current_timestamp)

                    else:

                        lying_stable_start = None

                       

                        if hundred.state == "HOLD":

                            if pelvis_status == "UNSTABLE":

                                border_color = (0, 0, 255)

                                trigger_voice_feedback("골반이 흔들립니다")

                                subtitle_text = "골반 고정하세요!"



                        res_badge, res_col, res_sub, res_voice, res_border = hundred.update(angle, current_timestamp)

                       

                        badge_text = res_badge

                        badge_color = res_col

                        subtitle_text = res_sub

                        if res_border: border_color = res_border

                        if res_voice: speak_message(res_voice)



                    recorded_data_log.append(log_entry)



                else:

                    lying_stable_start = None

                    if hundred.state != "IDLE" and hundred.state != "FINISHED":

                        hundred.reset()

                        speak_message("자세가 풀렸습니다.")

                   

                    badge_text = "인식 불가"

                    subtitle_text = "전신이 보이게 누워주세요"

           

            # UI 그리기

            frame = draw_ui_text(frame, badge_text, (20, 20), 24, bg_color=badge_color, align="left")

            if subtitle_text:

                frame = draw_ui_text(frame, subtitle_text, (w//2, h-50), 20, bg_color=(0,0,0), align="center")

            if border_color:

                cv2.rectangle(frame, (0,0), (w, h), border_color, 15)



            # 화면 병합

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

            dark_brown = (101, 67, 33)



            # 2. 텍스트 크기 계산

            text = "자세를 취해주세요"

            bbox = draw.textbbox((0, 0), text, font=font_guide)

            text_w = bbox[2] - bbox[0]

            text_h = bbox[3] - bbox[1]



            # 3. 위치 계산 (왼쪽 이미지 영역의 상단 중앙)

            if guide_img_resized is not None:

                w_guide = guide_img_resized.shape[1]

                x_pos = (w_guide - text_w) // 2

            else:

                # 가이드 이미지가 없을 경우 전체 화면의 1/4 지점

                h_final, w_final = final_display.shape[:2]

                x_pos = (w_final // 4) - (text_w // 2)



            y_pos = 50



            # 4. 텍스트 그리기 (진한 갈색, Bold 처리)

            draw.text((x_pos, y_pos), text, font=font_guide, fill=dark_brown, stroke_width=1, stroke_fill=dark_brown)

           

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

            csv_filename = f"Hundred_Log_{timestamp_str}.csv"

            pd.DataFrame(recorded_data_log).to_csv(csv_filename, index=False)

            print(f"[저장 완료] 로그 파일: {csv_filename}")



if __name__ == "__main__":

    run_hundred_coach()