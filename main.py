import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

# ==========================================
# 설정: 각 파일의 실제 경로를 입력하세요
# ==========================================
# (현재 main.py가 있는 위치를 기준으로 한 상대 경로)
SCRIPTS = {
    "lunge": "lunge/lunge_main_final.py",
    "mermaid": "mermaid/mermaid_main_final.py",
    "hundred": "hundred/hundred_main_final.py"
}

class PilatesLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Pilates Coach Launcher")
        self.root.geometry("400x500")
        self.root.configure(bg="#f0f0f0")

        # 제목
        title_label = tk.Label(root, text="AI Pilates Coach", font=("Malgun Gothic", 20, "bold"), bg="#f0f0f0")
        title_label.pack(pady=40)

        # 버튼 스타일
        btn_font = ("Malgun Gothic", 12)
        btn_bg = "#ffffff"
        btn_width = 25
        btn_height = 2

        # 1. 런지 버튼
        btn_lunge = tk.Button(root, text="① 런지 (Lunge) 신체평가", font=btn_font, width=btn_width, height=btn_height,
                              bg="#E1F5FE", command=lambda: self.run_script("lunge"))
        btn_lunge.pack(pady=10)

        # 2. 헌드레드 버튼
        btn_hundred = tk.Button(root, text="② 헌드레드 (Hundred) 코칭", font=btn_font, width=btn_width, height=btn_height,
                                bg="#E8F5E9", command=lambda: self.run_script("hundred"))
        btn_hundred.pack(pady=10)

        # 3. 머메이드 버튼
        btn_mermaid = tk.Button(root, text="③ 머메이드 (Mermaid) 코칭", font=btn_font, width=btn_width, height=btn_height,
                                bg="#FFF3E0", command=lambda: self.run_script("mermaid"))
        btn_mermaid.pack(pady=10)

        # 종료 버튼
        btn_exit = tk.Button(root, text="종료 (Exit)", font=btn_font, width=btn_width, height=btn_height,
                             bg="#FFEBEE", fg="red", command=root.quit)
        btn_exit.pack(pady=30)

        # 상태 표시줄
        self.status_label = tk.Label(root, text="운동을 선택해주세요.", font=("Malgun Gothic", 9), bg="#f0f0f0", fg="gray")
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    def run_script(self, key):
        script_path = SCRIPTS.get(key)
        
        # 파일 존재 여부 확인
        if not os.path.exists(script_path):
            messagebox.showerror("오류", f"파일을 찾을 수 없습니다:\n{script_path}")
            return

        self.status_label.config(text=f"실행 중: {script_path}...")
        self.root.update()

        try:
            # subprocess를 사용하여 별도의 파이썬 프로세스로 실행
            # 이렇게 하면 메모리, 카메라 자원이 완전히 독립됩니다.
            subprocess.run([sys.executable, script_path], check=True)
            
            self.status_label.config(text="운동 완료. 대기 중...")
        except subprocess.CalledProcessError as e:
            self.status_label.config(text="실행 중 오류 발생")
            messagebox.showerror("실행 오류", f"프로그램 실행 중 오류가 발생했습니다.\n{e}")
        except Exception as e:
            self.status_label.config(text="오류 발생")
            messagebox.showerror("오류", f"알 수 없는 오류:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PilatesLauncher(root)
    root.mainloop()