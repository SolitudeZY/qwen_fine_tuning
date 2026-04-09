import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import csv
import os

CSV_PATH = "/home/fs-ai/llama-qwen/outputs/all_fences_for_review.csv"

class ReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("围栏图片违规审查工具")
        self.root.geometry("1000x800")
        
        self.data = []
        self.current_idx = 0
        
        self.load_csv()
        self.setup_ui()
        self.show_current_image()

    def load_csv(self):
        if not os.path.exists(CSV_PATH):
            messagebox.showerror("错误", f"找不到CSV文件: {CSV_PATH}")
            self.root.destroy()
            return
            
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            self.header = next(reader)
            self.data = list(reader)

    def save_csv(self):
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(self.data)

    def setup_ui(self):
        # 顶部信息栏
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(fill=tk.X, pady=10)
        
        self.progress_label = tk.Label(self.info_frame, text="", font=("Arial", 14))
        self.progress_label.pack(side=tk.LEFT, padx=20)
        
        self.status_label = tk.Label(self.info_frame, text="", font=("Arial", 14, "bold"))
        self.status_label.pack(side=tk.RIGHT, padx=20)
        
        self.file_label = tk.Label(self.info_frame, text="", font=("Arial", 10), fg="gray")
        self.file_label.pack(side=tk.TOP, pady=5)

        # 图片显示区域
        self.image_label = tk.Label(self.root)
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # 底部按钮栏
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(fill=tk.X, pady=20)
        
        tk.Button(self.btn_frame, text="上一张 (←)", command=self.prev_image, font=("Arial", 12), width=15).pack(side=tk.LEFT, padx=20)
        
        # 核心标记按钮
        self.btn_y = tk.Button(self.btn_frame, text="违规 (Y)", command=lambda: self.mark("Y"), font=("Arial", 14, "bold"), bg="#ffcccc", width=15)
        self.btn_y.pack(side=tk.LEFT, padx=10)
        
        self.btn_n = tk.Button(self.btn_frame, text="合规 (N)", command=lambda: self.mark("N"), font=("Arial", 14, "bold"), bg="#ccffcc", width=15)
        self.btn_n.pack(side=tk.LEFT, padx=10)
        
        self.btn_skip = tk.Button(self.btn_frame, text="跳过 / 不确定", command=lambda: self.mark("Review"), font=("Arial", 12), width=15)
        self.btn_skip.pack(side=tk.LEFT, padx=10)
        
        tk.Button(self.btn_frame, text="下一张 (→)", command=self.next_image, font=("Arial", 12), width=15).pack(side=tk.RIGHT, padx=20)

        # 绑定快捷键
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("y", lambda e: self.mark("Y"))
        self.root.bind("Y", lambda e: self.mark("Y"))
        self.root.bind("n", lambda e: self.mark("N"))
        self.root.bind("N", lambda e: self.mark("N"))

    def show_current_image(self):
        if not self.data:
            return
            
        row = self.data[self.current_idx]
        img_path = row[0]
        fence_types = row[2]
        current_mark = row[3]
        
        # 更新进度和状态信息
        self.progress_label.config(text=f"进度: {self.current_idx + 1} / {len(self.data)}")
        self.file_label.config(text=f"文件: {os.path.basename(img_path)} | 包含标签: {fence_types}")
        
        if current_mark == "Y":
            self.status_label.config(text="当前标记: 🔴 违规 (Y)", fg="red")
        elif current_mark == "N":
            self.status_label.config(text="当前标记: 🟢 合规 (N)", fg="green")
        elif current_mark == "Review":
            self.status_label.config(text="当前标记: 🟡 待定 (Review)", fg="orange")
        else:
            self.status_label.config(text="当前标记: ⚪ 未标记", fg="gray")

        # 加载并缩放图片
        try:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                # 计算缩放比例以适应窗口
                display_size = (900, 600)
                img.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo # 保持引用防止被垃圾回收
            else:
                self.image_label.config(image="", text="[图片文件不存在]")
        except Exception as e:
            self.image_label.config(image="", text=f"[加载图片出错: {e}]")

    def mark(self, value):
        if not self.data: return
        self.data[self.current_idx][3] = value
        self.save_csv()
        self.next_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_image()

    def next_image(self):
        if self.current_idx < len(self.data) - 1:
            self.current_idx += 1
            self.show_current_image()
        else:
            messagebox.showinfo("完成", "已经是最后一张图片了！")
            self.show_current_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ReviewApp(root)
    root.mainloop()