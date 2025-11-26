# [새싹_성북 헬스케어분야 서비스 기획자 부트캠프 4기] Final Project 1Team 
## **HOMElates : YOLO11/센서 기반 실시간 AI필라테스 자세 교정 서비스 기획**
팀명 : 필라텔레토비  (강주현, 김민호, 김한주, 신의진, 조윤아, 황규원)

## 1. 프로젝트 소개
HOMElates는 사용자 운동 자세와 정확성을 체계적으로 분석·평가하기 위해, 영상·센서 데이터 소스를 통합하고 최적의 머신러닝 모델을 적용한 데이터 기반 서비스로 설계되었습니다.

- 일정: 2025.11.10 - 2025.11.26
- 사용 데이터: 필라테스 영상 데이터, 신체 좌표 및 각도 데이터, 골반 불균형 데이터
- 사용 모델/기기: YOLO, MPU6050 센서
- 프로세스:
  
<img width="1599" height="409" alt="image" src="https://github.com/user-attachments/assets/e04e515b-3aa3-4c2c-9c85-146b68d66def" />


## 2. 진행 내용 상세
필라테스 전문가 영상 데이터를 분석하여 필라테스 동작별 정상 동작 범위를 라벨링하였습니다.
데이터 라벨에 따른 사용자 분석 데이터를 토대로 실시간 음성, 자막 피드백과 평가 점수를 제공하는 서비스를 구현하였습니다.


## 시작가이드

### 📂 디렉토리 구조 (Directory Structure)

```bash
├── 📁 hundred/                  # 헌드레드(Hundred) 동작 교정 모듈
│   ├── hundred_main_final.py    # 헌드레드 메인 실행 파일
│   └── reference.png            # 헌드레드 참고 자세 이미지
│
├── 📁 lunge/                    # 런지(Lunge) 자세 측정 모듈
│   ├── lunge.png                # 런지 참고 자세 이미지
│   └── lunge_main_final.py      # 최종 런지 분석 로직
│
├── 📁 mermaid/                  # 머메이드(Mermaid)자세 교정 모듈
│   ├── mermaid_main_final.py    # 머메이드 메인 실행 파일
│   └── reference.png            # 머메이드 참고 자세
│
├── main.py                      # main 실행 파일(런지, 머메이드, 헌드레드 실행)
│
├── .gitignore                   # Git 업로드 제외 설정
└── README.md                    # 프로젝트 설명서
```
### 설치
```bash
# 레포지토리 클론
git clone [https://github.com/사용자아이디/레포지토리명.git](https://github.com/사용자아이디/레포지토리명.git)

# 패키지 설치
pip install ultralytics opencv-python pandas

# 실행
python main.py
```



## Stacks

**Environment**
<br>
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
<br>
**Communication**
<br>
![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)
<br>
**Sensor**
<br>
![Raspberry Pi](https://img.shields.io/badge/-Raspberry_Pi-C51A4A?style=for-the-badge&logo=Raspberry-Pi)
<br>

