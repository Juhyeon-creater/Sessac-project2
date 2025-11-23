import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import joblib
from collections import deque
import pyttsx3
import threading
import queue
import time
import os

# --- 설정 및 상수 ---
MODEL_PATH = 'lunge/lunge_lstm_model.h5'
SCALER_PATH = 'lunge/scaler.pkl'
GUIDE_IMAGE_PATH = 'lunge/lunge.png'
SEQUENCE_LENGTH = 30

# [중요] 인식 기준 완화 (기존 110 -> 125)
# 숫자가 180(서있음) -> 90(앉음)으로 갈수록 낮아집니다.
# 125도보다 낮아지면 "자세 잡음"으로 인정합니다.
LUNGE_KNEE_THRESHOLD = 125 
HOLD_DURATION = 2.0

# --- 안전한 음성 출력 (Queue) ---
speech_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        text = speech_queue.get()
        if text is None: break
        try:
            engine.say(text)
            engine.runAndWait()
        except: pass
        speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)

# --- 유틸리티 ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

class LungeCoach:
    def __init__(self):
        self.lstm_model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.yolo = YOLO('yolo11s-pose.pt')
        
        self.state = "INIT"
        self.buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.state_start_time = time.time()
        self.hold_start_time = None
        self.current_set = 1
        self.total_sets = 3
        self.score_history = [] 
        
        # 이미지 로드 (나중에 웹캠 높이에 맞춰 리사이즈함)
        self.guide_img_raw = None
        if os.path.exists(GUIDE_IMAGE_PATH):
            self.guide_img_raw = cv2.imread(GUIDE_IMAGE_PATH)
            print("✅ 가이드 이미지 로드 성공")
        else:
            print("⚠️ 가이드 이미지가 없습니다. 검은 화면으로 대체됩니다.")

    def get_lstm_score(self):
        if len(self.buffer) < SEQUENCE_LENGTH: return 0.0
        input_data = np.expand_dims(list(self.buffer), axis=0)
        prediction = self.lstm_model.predict(input_data, verbose=0)[0][0]
        return float((1 - prediction) * 100)

    def draw_report(self, canvas, cam_w):
        """종료 리포트 (오른쪽 웹캠 영역에만 그림)"""
        # 웹캠 영역의 시작 x좌표
        x_start = canvas.shape[1] - cam_w
        
        overlay = canvas.copy()
        # 오른쪽 영역 전체 어둡게
        cv2.rectangle(overlay, (x_start, 0), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        avg_score = sum(self.score_history) / len(self.score_history) if self.score_history else 0
        
        cx = x_start + 50 # 텍스트 시작점
        cv2.putText(canvas, "WORKOUT COMPLETE", (cx, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(canvas, f"Avg Score: {avg_score:.1f}", (cx, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        y_pos = 200
        for i, score in enumerate(self.score_history):
            if i < 6:
                text = f"Set {(i//2)+1} {'L' if i%2==0 else 'R'}: {score:.1f}"
                cv2.putText(canvas, text, (cx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                y_pos += 40
        return canvas

    def process_frame(self, frame):
        # 1. 웹캠 프레임 기본 정보
        h, w = frame.shape[:2]

        # 2. 가이드 이미지 리사이즈 (웹캠 높이 h에 맞춤)
        if self.guide_img_raw is not None:
            gh, gw = self.guide_img_raw.shape[:2]
            new_gw = int(gw * (h / gh))
            guide_resized = cv2.resize(self.guide_img_raw, (new_gw, h))
        else:
            # 이미지 없으면 그냥 검은색 빈 공간 생성
            guide_resized = np.zeros((h, 300, 3), dtype=np.uint8)
            new_gw = 300

        # 3. YOLO 추론 (웹캠 화면에 대해서만 수행)
        results = self.yolo(frame, verbose=False)
        current_time = time.time()
        
        # UI용 변수 (오른쪽 화면 기준 좌표)
        status_text = f"{self.state}"
        
        # 스켈레톤 그리기 및 데이터 추출
        if results[0].keypoints and len(results[0].keypoints) > 0:
            kpts = results[0].keypoints.xyn[0].cpu().numpy()
            l_hip, l_knee, l_ankle = kpts[11], kpts[13], kpts[15]
            r_hip, r_knee, r_ankle = kpts[12], kpts[14], kpts[16]

            # 감지 확인
            if l_knee[0] > 0 and r_knee[0] > 0:
                angle_l = calculate_angle(l_hip, l_knee, l_ankle)
                angle_r = calculate_angle(r_hip, r_knee, r_ankle)
                ankle_dist = abs(l_ankle[0] - r_ankle[0])
                
                # [디버깅] 현재 각도 화면에 표시 (인식 안될 때 확인용)
                # 왼쪽 무릎 근처에 각도 표시
                lx, ly = int(l_knee[0]*w), int(l_knee[1]*h)
                rx, ry = int(r_knee[0]*w), int(r_knee[1]*h)
                
                # 각도 텍스트 색상 (기준 충족하면 초록, 아니면 빨강)
                col_l = (0, 255, 0) if angle_l < LUNGE_KNEE_THRESHOLD else (0, 0, 255)
                col_r = (0, 255, 0) if angle_r < LUNGE_KNEE_THRESHOLD else (0, 0, 255)
                
                cv2.putText(frame, f"{int(angle_l)}deg", (lx+10, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_l, 2)
                cv2.putText(frame, f"{int(angle_r)}deg", (rx+10, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_r, 2)

                # 데이터 저장
                raw = np.array([[angle_l, angle_r, l_hip[1], r_hip[1], ankle_dist]])
                scaled = self.scaler.transform(raw)
                self.buffer.append(scaled[0])

                # 뼈대 그리기
                for kp in [l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]:
                    x, y = int(kp[0]*w), int(kp[1]*h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # --- 상태 머신 로직 ---
                if self.state == "INIT":
                    if current_time - self.state_start_time > 1.0:
                        speak("평가를 시작합니다. 시범 영상처럼 측면으로 서주세요.")
                        self.state = "SIDE_CHECK"

                elif self.state == "SIDE_CHECK":
                    if ankle_dist > 0.05:
                        speak("왼다리를 앞으로 굽혀주세요.")
                        self.state = "LEFT_READY"
                        self.buffer.clear()
                    else:
                        cv2.putText(frame, "Stand Sideways", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # [왼쪽 다리]
                elif self.state == "LEFT_READY":
                    # 기준 각도보다 작아지면(굽혀지면) 카운트 시작
                    if angle_l < LUNGE_KNEE_THRESHOLD:
                        self.state = "LEFT_HOLD"
                        self.hold_start_time = current_time
                        speak("버티세요!")

                elif self.state == "LEFT_HOLD":
                    elapsed = current_time - self.hold_start_time
                    bar_width = int((elapsed / HOLD_DURATION) * 200)
                    # 타이머 바 그리기
                    cv2.rectangle(frame, (50, 100), (50 + bar_width, 130), (0, 255, 255), -1)
                    cv2.rectangle(frame, (50, 100), (250, 130), (255, 255, 255), 2)
                    
                    if angle_l > LUNGE_KNEE_THRESHOLD + 20: # 자세 풀림 (여유 20도)
                        self.state = "LEFT_READY"
                        speak("풀렸습니다.")
                    
                    elif elapsed >= HOLD_DURATION:
                        score = self.get_lstm_score()
                        if score >= 50:
                            self.score_history.append(score)
                            speak("좋습니다. 오른 다리.")
                            self.state = "RIGHT_READY"
                        else:
                            speak("다시 하세요.")
                            self.state = "LEFT_READY"
                        self.buffer.clear()

                # [오른쪽 다리]
                elif self.state == "RIGHT_READY":
                    if angle_r < LUNGE_KNEE_THRESHOLD:
                        self.state = "RIGHT_HOLD"
                        self.hold_start_time = current_time
                        speak("버티세요!")

                elif self.state == "RIGHT_HOLD":
                    elapsed = current_time - self.hold_start_time
                    bar_width = int((elapsed / HOLD_DURATION) * 200)
                    cv2.rectangle(frame, (50, 100), (50 + bar_width, 130), (0, 255, 255), -1)
                    cv2.rectangle(frame, (50, 100), (250, 130), (255, 255, 255), 2)

                    if angle_r > LUNGE_KNEE_THRESHOLD + 20:
                        self.state = "RIGHT_READY"
                        speak("풀렸습니다.")

                    elif elapsed >= HOLD_DURATION:
                        score = self.get_lstm_score()
                        if score >= 50:
                            self.score_history.append(score)
                            if self.current_set < self.total_sets:
                                speak(f"{self.current_set}세트 끝. 다음 세트.")
                                self.current_set += 1
                                self.state = "LEFT_READY"
                            else:
                                speak("운동 완료.")
                                self.state = "END"
                        else:
                            speak("다시 하세요.")
                            self.state = "RIGHT_READY"
                        self.buffer.clear()

        # 4. 화면 결합 (Stitch)
        # [가이드 이미지] | [웹캠 영상]
        canvas = np.hstack((guide_resized, frame))

        # 5. 상태 텍스트는 전체 화면 중앙 상단에 표시
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
        
        info_text = f"Set {self.current_set}/{self.total_sets} | {self.state}"
        if self.state == "END":
            # 종료 시 리포트 오버레이
            canvas = self.draw_report(canvas, w)
        else:
            # 평소엔 상태 표시
            cv2.putText(canvas, info_text, (new_gw + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        return canvas

def main():
    coach = LungeCoach()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # 거울 모드
        
        # process_frame에서 이미지가 합쳐져서 나옴 (canvas)
        final_view = coach.process_frame(frame)
        
        cv2.imshow('AI Lunge Coach (Split View)', final_view)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()