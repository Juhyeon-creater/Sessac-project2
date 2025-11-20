# -------------------------------------------------
# Hundred, Mermaid 자세의 상관계수 분석
#    - YOLO Angle ↔ Pelvis Roll Diff 상관계수 분석
#    - Pelvis Down ↔ YOLO Bad 분류 상관관계
#    - 포트폴리오 제출용 코드 (주석 상세)
# -------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# 1. CSV 로드
# --------------------------------------------
csv_path = "/content/drive/MyDrive/shared_googledrive(Sessac Final Project)/video/머메이드/HJ_Hundred_Mermaid_20251120.csv"
df = pd.read_csv(csv_path)

# --------------------------------------------
# 2. 결측 처리 (Pelvis 데이터 없는 행 제거)
# --------------------------------------------
df = df.dropna(subset=["MPU_Pelvis_Status", "MPU_Roll_Diff", "Angle"])

# --------------------------------------------
# 3. Pelvis Down Binary 변환
#    LEFT DOWN 또는 RIGHT DOWN → 1
#    LEVEL → 0
# --------------------------------------------
def pelvis_to_binary(x):
    x = str(x).upper()
    if "LEFT DOWN" in x or "RIGHT DOWN" in x:
        return 1
    return 0

df["Pelvis_Down"] = df["MPU_Pelvis_Status"].apply(pelvis_to_binary)

# --------------------------------------------
# 4. YOLO Bad 여부 (GOOD=0, BAD=1)
# --------------------------------------------
df["YOLO_Bad"] = df["Is_Good"].apply(lambda x: 0 if str(x).upper()=="TRUE" or x==1 else 1)

# --------------------------------------------
# 5. 핵심 상관관계 분석
#    (1) YOLO Angle ↔ Pelvis Roll Diff
#    (2) YOLO BAD ↔ Pelvis Down
# --------------------------------------------
corr_angle = df["Angle"].corr(df["MPU_Roll_Diff"])       # 회귀형 상관
corr_binary = df["YOLO_Bad"].corr(df["Pelvis_Down"])     # 이진 분류 상관

print("-----------------------------------")
print("Correlation Analysis Result")
print("-----------------------------------")
print(f"1) YOLO Angle ↔ Pelvis Roll Diff : {corr_angle:.3f}")
print(f"2) YOLO BAD ↔ Pelvis Down        : {corr_binary:.3f}")
print("-----------------------------------")

# --------------------------------------------
# 6. 그래프 그리기 (포트폴리오용 시각화)
# --------------------------------------------

# 6-1) Angle vs Pelvis Diff 산점도 + 회귀선
plt.figure(figsize=(8,6))
sns.regplot(x=df["Angle"], y=df["MPU_Roll_Diff"], line_kws={"color":"red"})
plt.title("YOLO Angle vs Pelvis Roll Diff (Correlation)")
plt.xlabel("YOLO Leg Angle (degree)")
plt.ylabel("Pelvis Roll Difference (degree)")
plt.grid(True)
plt.show()
