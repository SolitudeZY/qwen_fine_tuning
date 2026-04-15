from flask import Flask, render_template_string, request, jsonify, send_file
import csv
import os

app = Flask(__name__)

CSV_PATH = "/home/fs-ai/llama-qwen/outputs/all_fences_for_review.csv"
data = []
header = []

def load_csv():
    global data, header
    if not os.path.exists(CSV_PATH):
        return False
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return True

def save_csv():
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>围栏图片审查工具 (Web)</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #f5f5f5; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .header { display: flex; justify-content: space-between; margin-bottom: 10px; }
        .image-box { margin: 20px 0; min-height: 500px; display: flex; justify-content: center; align-items: center; background: #eee; }
        img { max-width: 100%; max-height: 600px; object-fit: contain; }
        .controls { margin-top: 20px; }
        button { font-size: 18px; padding: 10px 20px; margin: 0 10px; cursor: pointer; border: none; border-radius: 5px; }
        .btn-y { background-color: #ff4d4d; color: white; font-weight: bold; }
        .btn-n { background-color: #4CAF50; color: white; font-weight: bold; }
        .btn-nav { background-color: #008CBA; color: white; }
        .status { font-size: 20px; font-weight: bold; margin-bottom: 10px; }
        .info { color: #555; font-size: 14px; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2 id="progress">进度: 0 / 0</h2>
            <div id="status" class="status">当前标记: 未知</div>
        </div>
        <div class="info" id="info">文件名: - | 包含标签: -</div>
        
        <div class="image-box">
            <img id="image" src="" alt="图片加载中...">
        </div>

        <div class="controls">
            <button class="btn-nav" onclick="navigate(-1)">← 上一张</button>
            <button class="btn-y" onclick="mark('Y')">违规 (Y)</button>
            <button class="btn-n" onclick="mark('N')">合规 (N)</button>
            <button class="btn-nav" onclick="mark('Review')">跳过 (待定)</button>
            <button class="btn-nav" onclick="navigate(1)">下一张 →</button>
        </div>
    </div>

    <script>
        let currentIdx = 0;
        let totalItems = 0;

        function loadData(idx) {
            fetch(`/api/data?idx=${idx}`)
                .then(res => res.json())
                .then(data => {
                    if (data.error) {
                        alert(data.error);
                        return;
                    }
                    currentIdx = data.idx;
                    totalItems = data.total;
                    
                    document.getElementById('progress').innerText = `进度: ${currentIdx + 1} / ${totalItems}`;
                    document.getElementById('info').innerText = `文件名: ${data.filename} | 包含标签: ${data.labels}`;
                    
                    // 强制刷新图片缓存
                    document.getElementById('image').src = `/api/image?path=${encodeURIComponent(data.image_path)}&t=${new Date().getTime()}`;
                    
                    const statusEl = document.getElementById('status');
                    if (data.mark === 'Y') {
                        statusEl.innerText = "当前标记: 🔴 违规 (Y)";
                        statusEl.style.color = "red";
                    } else if (data.mark === 'N') {
                        statusEl.innerText = "当前标记: 🟢 合规 (N)";
                        statusEl.style.color = "green";
                    } else if (data.mark === 'Review') {
                        statusEl.innerText = "当前标记: 🟡 待定";
                        statusEl.style.color = "orange";
                    } else {
                        statusEl.innerText = "当前标记: ⚪ 未标记";
                        statusEl.style.color = "gray";
                    }
                });
        }

        function mark(value) {
            fetch('/api/mark', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ idx: currentIdx, mark: value })
            }).then(() => navigate(1));
        }

        function navigate(step) {
            let nextIdx = currentIdx + step;
            if (nextIdx < 0) nextIdx = 0;
            if (nextIdx >= totalItems) {
                alert("已经是最后一张图片了！");
                return;
            }
            loadData(nextIdx);
        }

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.key === 'y' || e.key === 'Y') mark('Y');
            else if (e.key === 'n' || e.key === 'N') mark('N');
            else if (e.key === 'ArrowLeft') navigate(-1);
            else if (e.key === 'ArrowRight') navigate(1);
        });

        // 初始化
        window.onload = () => loadData(0);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    if not load_csv():
        return "找不到 CSV 文件，请先运行提取脚本！"
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/data")
def get_data():
    idx = int(request.args.get("idx", 0))
    if idx < 0 or idx >= len(data):
        return jsonify({"error": "索引越界"})
    
    row = data[idx]
    return jsonify({
        "idx": idx,
        "total": len(data),
        "image_path": row[0],
        "filename": os.path.basename(row[0]),
        "labels": row[2],
        "mark": row[3] if len(row) > 3 else ""
    })

@app.route("/api/image")
def get_image():
    img_path = request.args.get("path")
    if not img_path or not os.path.exists(img_path):
        return "Image not found", 404
    return send_file(img_path, mimetype='image/jpeg')

@app.route("/api/mark", methods=["POST"])
def mark_data():
    req = request.json
    idx = req.get("idx")
    mark_value = req.get("mark")
    
    if 0 <= idx < len(data):
        # 确保行有足够的列
        while len(data[idx]) < 4:
            data[idx].append("")
        data[idx][3] = mark_value
        save_csv()
        return jsonify({"success": True})
    return jsonify({"error": "Invalid index"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
