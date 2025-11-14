import pandas as pd

df = pd.read_csv('dataset\LegPull\LegPull -_cam1\keypoints_필라테스_가산_A__Leg Pull_고급_actorP061_20220829_11.23.50_CAM_1.csv')
#print(df)


from ultralytics import YOLO
import cv2 


# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n-pose.pt")


# 2. 웹캠 켜기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. YOLO 모델로 추론 (자세 추정)
    # results 안에 관절 좌표(Keypoints)가 들어있습니다.
    results = model(frame, verbose = False)

    # 4. 화면에 결과 그리기 (뼈대 표시)
    annotated_frame = results[0].plot()

    # 5. 화면 출력
    cv2.imshow('YOLO11 Pose', annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()