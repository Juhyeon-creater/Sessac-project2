# [새싹 헬스케어 서비스 기획자 부트캠프 4기] Final Project - Team 2
## **HOMElates : YOLO11/센서 기반 실시간 AI필라테스 자세 교정 서비스 기획**
팀명 : 필라텔레토비  (강주현, 김민호, 김한주, 신의진, 조윤아, 황규원)

## 1. 프로젝트 소개
HOMElates는 홈트레이닝 환경에서 필라테스 동작을 실시간으로 분석하고 교정해주는 AI 기반 자세 교정 시스템입니다.
YOLO11 Pose Estimation 모델과 자이로/가속도 센서를 결합하여 정확한 자세 피드백을 제공합니다.

- 일정: 2025.11.01 - 2025.11.26 (4주)
- 사용 데이터: 필라테스 영상 데이터, 신체 좌표 및 각도 데이터, 골반 불균형 데이터
- 사용 모델/기기: YOLO, MPU6050 센서
- 프로세스:
  
<img width="1599" height="409" alt="image" src="https://github.com/user-attachments/assets/e04e515b-3aa3-4c2c-9c85-146b68d66def" />

- 프로젝트 목표
  - 실시간 필라테스 동작 인식, 분석, 음성 코칭
  - 골반 각도 측정을 통한 정밀한 자세 교정
  - 초보자도 쉽게 사용할 수 있는 직관적인 인터페이스
 
- 담당 역할
  - YOLO + 자이로/가속도 센서 융합
  - 자이로/가속도 센서 회로 설계 및 브레드보드 구현
  - 산점도, Line graph 기반 상관관계 분석
  - LSTM 기반 모델 성능 평가
  - 최종 발표
 
- 주요 기능
  - 실시간 자세 인식
    - YOLO11 Pose Estimation: 17개 신체 키포인트 실시간 추적
    - 프레임 처리 속도: 평균 30 FPS
    - 지원 동작: Hundred (헌드레드), Lunge (런지), Mermaid (머메이드)
  - 골반 각도 측정
    - MPU6050 센서: 6축 가속도/자이로 센서로 정밀한 골반 각도 측정
    - 실시간 피드백: 각도 편차에 따른 즉각적인 교정 가이드
    - 시각적 표시: 화면에 각도 수치 및 교정 방향 표시
  - 3가지 필라테스 동작 지원
    - Hundred (헌드레드):	복부 코어 강화 동작-골반 중립 자세, 다리 각도, 상체 안정성
    - Lunge (런지):	하체 근력 강화 동작-무릎 정렬, 골반 기울기, 상체 균형
    - Mermaid (머메이드):	측면 스트레칭 동작-척추 정렬, 골반 안정성, 팔 각도

## 2. 진행 내용 상세
필라테스 전문가 영상 데이터를 분석하여 필라테스 동작별 정상 동작 범위를 라벨링하였습니다.
데이터 라벨에 따른 사용자 분석 데이터를 토대로 실시간 음성, 자막 피드백과 평가 점수를 제공하는 서비스를 구현하였습니다.


## 3. 시작가이드

### 📂 디렉토리 구조 (Directory Structure)

```bash
├── 📁 Analysis/                        # 분석 및 연구 노트북
│   ├── JH_hundred_pose_yolo_pelvis_correlation.ipynb
│   ├── JH_hundred_pose_yolo_pelvis_evaluation.ipynb
│   ├── YOLO Pose Normalization & Preprocessing Pipeline.ipynb
│   └── YOLO_applied_video.ipynb
│
├── 📁 hundred/                         # 헌드레드(Hundred) 동작 교정 모듈
│   ├── hundred_main_final.py           # 헌드레드 메인 실행 파일
│   └── reference.png                   # 헌드레드 참고 자세 이미지
│
├── 📁 lunge/                           # 런지(Lunge) 자세 측정 모듈
│   ├── lunge_main_final.py             # 런지 메인 실행 파일
│   └── lunge.png                       # 런지 참고 자세 이미지
│
├── 📁 mermaid/                         # 머메이드(Mermaid) 자세 교정 모듈
│   ├── mermaid_main_final.py           # 머메이드 메인 실행 파일
│   └── reference.png                   # 머메이드 참고 자세 이미지
│
├── 📄 main.py                          # 메인 실행 파일 (전체 시스템 통합)
├── 📄 requirements.txt                 # 의존성 패키지 목록
├── 📄 .gitignore                       # Git 제외 파일 설정
└── 📄 README.md                        # 프로젝트 설명서
```
### ⬇️설치
```bash
# 레포지토리 클론
git clone https://github.com/Juhyeon-creater/Sessac-project2.git

# 패키지 설치
pip install -r requirements.txt

# 실행 (원하는 모드 선택)
python main.py
```



## Stacks

**Environment**
<br>
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
<br>
**Development & AI**
<br>
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![YOLO](https://img.shields.io/badge/YOLO-111F68?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
<br>
**Communication**
<br>
![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)
![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)
![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)
<br>
**Hardware**
<br>
![Raspberry Pi](https://img.shields.io/badge/-Raspberry_Pi-C51A4A?style=for-the-badge&logo=Raspberry-Pi)
<img src="https://img.shields.io/badge/Sensor-MPU6050-blue?style=for-the-badge" alt="MPU6050">
<br>

