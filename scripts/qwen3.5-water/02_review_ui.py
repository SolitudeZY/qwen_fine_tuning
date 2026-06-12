"""
环水保标注人工审查 Web UI
读取 data/annotate_cache_water.json，支持：
  - 检查图片
  - 修改水体状态 (normal / polluted)
  - 修改 API 生成的描述 (description)
  - 审查结果写回 annotate_cache_water.json
  - 导出为 Qwen3.5-VL 微调所需的 JSONL 格式

用法：
  python scripts/qwen3.5-water/02_review_ui.py
  # 本机 SSH 端口转发：ssh -L 5002:localhost:5002 user@server
  # 浏览器打开 http://localhost:5002
"""

import copy
import json
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file

ROOT = Path(__file__).parent.parent.parent
CACHE_FILE = ROOT / "data" / "annotate_cache_water.json"
OUTPUT_JSONL = ROOT / "data" / "环水保_labeled" / "water_finetune_reviewed.jsonl"

app = Flask(__name__)
cache: dict = {}
ordered_keys: list[str] = []

def load_cache():
    global cache, ordered_keys
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}
    
    # 排序：未审查的排前面
    ordered_keys = sorted(
        cache.keys(),
        key=lambda k: 1 if cache[k].get("reviewed") else 0
    )

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def filter_keys(mode: str) -> list[str]:
    if mode == "unreviewed":
        return [k for k in ordered_keys if not cache[k].get("reviewed")]
    if mode == "reviewed":
        return [k for k in ordered_keys if cache[k].get("reviewed")]
    if mode == "polluted":
        return [k for k in ordered_keys if cache[k].get("water_status") == "polluted"]
    if mode == "normal":
        return [k for k in ordered_keys if cache[k].get("water_status") == "normal"]
    return list(ordered_keys)

HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>环水保标注审查工具</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Microsoft YaHei", Arial, sans-serif; background: #1a1a2e; color: #eee; }
.layout { display: flex; height: 100vh; }
.sidebar { width: 350px; min-width: 350px; background: #16213e; padding: 16px; overflow-y: auto; border-right: 1px solid #0f3460; flex-shrink: 0; }
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
.toolbar { background: #0f3460; padding: 10px 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; flex-shrink: 0; }
.image-area { flex: 1; overflow: auto; display: flex; justify-content: center; align-items: center; padding: 16px; background: #0d0d1a; }
#bgImg { max-width: 100%; max-height: 100%; object-fit: contain; }
h2 { color: #e94560; margin-bottom: 12px; font-size: 18px; }
.meta-row { font-size: 14px; color: #a8b2d8; margin: 8px 0; }
.meta-row span { color: #eee; font-weight: bold; }
select, textarea { background: #1a1a2e; color: #eee; border: 1px solid #0f3460; border-radius: 4px; padding: 8px; font-size: 14px; width: 100%; }
select:focus, textarea:focus { outline: none; border-color: #e94560; }
button { cursor: pointer; border: none; border-radius: 4px; padding: 8px 16px; font-size: 14px; font-weight: bold; transition: opacity .2s; }
button:hover { opacity: 0.85; }
.btn-success { background: #4caf50; color: white; } 
.btn-danger { background: #f44336; color: white; }
.btn-info { background: #2196f3; color: white; } 
.btn-warning { background: #ff9800; color: white; }
.progress { font-size: 14px; color: #a8b2d8; margin-bottom: 10px; }
.filter-bar { display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }
.filter-btn { background: #0f3460; color: #a8b2d8; padding: 6px 12px; font-size: 12px; border-radius: 12px; border: 1px solid #0f3460; }
.filter-btn.active { background: #e94560; color: white; border-color: #e94560; }
.divider { border: none; border-top: 1px solid #0f3460; margin: 16px 0; }
.kbd { background: #0f3460; border-radius: 3px; padding: 2px 6px; font-size: 12px; font-family: monospace; }
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <h2>环水保标注审查</h2>
    <div class="filter-bar">
      <button class="filter-btn active" onclick="setFilter('all',this)">全部</button>
      <button class="filter-btn" onclick="setFilter('unreviewed',this)">未审查</button>
      <button class="filter-btn" onclick="setFilter('reviewed',this)">已审查</button>
      <button class="filter-btn" onclick="setFilter('polluted',this)">污染水体</button>
      <button class="filter-btn" onclick="setFilter('normal',this)">正常水体</button>
    </div>
    <div class="progress" id="progress">加载中...</div>
    <hr class="divider">
    <div class="meta-row">文件名：<br><span id="mFilename" style="word-break: break-all;">-</span></div>
    <div class="meta-row">审查状态：<span id="mReviewed">-</span></div>
    
    <div class="meta-row" style="margin-top: 16px;">水体状态 (Status)：</div>
    <select id="editStatus">
      <option value="normal">正常 (Normal)</option>
      <option value="polluted">污染 (Polluted)</option>
    </select>

    <div class="meta-row" style="margin-top: 16px;">描述 (Description)：</div>
    <textarea id="editDesc" rows="5" placeholder="请输入对图片的描述..."></textarea>

    <hr class="divider">

    <div style="display:flex;flex-direction:column;gap:12px;">
      <button class="btn-success" onclick="saveAndNext()">✅ 保存并下一张 <span class="kbd">S</span></button>
      <button class="btn-warning" onclick="excludeImage()">🗑 排除此图</button>
      <button class="btn-info" onclick="exportJsonl()" style="margin-top: 20px;">📦 导出为 JSONL 训练文件</button>
    </div>
    <hr class="divider">
    <div style="font-size:12px;color:#666;line-height:2">
      <span class="kbd">←</span><span class="kbd">→</span> 翻页 &nbsp;
      <span class="kbd">S</span> 保存并下一张
    </div>
  </div>

  <div class="main">
    <div class="toolbar">
      <button class="btn-info" onclick="navigate(-1)">← 上一张</button>
      <button class="btn-info" onclick="navigate(1)">下一张 →</button>
      <span id="toolbarInfo" style="font-size:14px;color:#a8b2d8"></span>
    </div>
    <div class="image-area">
      <img id="bgImg" src="" alt="加载中...">
    </div>
  </div>
</div>

<script>
let currentIdx = 0, currentFilter = 'all', ann = null;

function setFilter(f, btn) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentIdx = 0;
  loadData(0);
}

function loadData(idx) {
  fetch(`/api/data?idx=${idx}&filter=${currentFilter}`)
    .then(r => r.json())
    .then(data => {
      if (data.error) { document.getElementById('progress').textContent = data.error; return; }
      currentIdx = data.idx;
      ann = data.ann;
      document.getElementById('progress').textContent = `${data.idx+1} / ${data.total}`;
      document.getElementById('toolbarInfo').textContent = `${data.filename}`;
      document.getElementById('mFilename').textContent = data.filename;
      
      document.getElementById('mReviewed').textContent = ann.reviewed ? '✅ 已审查' : '⚪ 未审查';
      document.getElementById('editStatus').value = ann.water_status || 'normal';
      document.getElementById('editDesc').value = ann.description || '';
      
      document.getElementById('bgImg').src = `/api/image?idx=${data.idx}&filter=${currentFilter}&t=${Date.now()}`;
    });
}

function saveAndNext() {
  const payload = {
    idx: currentIdx, 
    filter: currentFilter,
    water_status: document.getElementById('editStatus').value,
    description: document.getElementById('editDesc').value
  };
  fetch('/api/save', { 
    method:'POST', 
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  }).then(() => navigate(1));
}

function excludeImage() {
  if (!confirm('确认排除此图？它将从 cache 中移除。')) return;
  fetch('/api/exclude', { 
    method:'POST', 
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ idx: currentIdx, filter: currentFilter })
  }).then(r=>r.json()).then(d => {
    if (d.error) { alert(d.error); return; }
    loadData(currentIdx);
  });
}

function exportJsonl() {
  if (!confirm('确认导出所有已审查的数据为 JSONL 格式？')) return;
  fetch('/api/export', { method:'POST' })
    .then(r=>r.json())
    .then(d => {
      if (d.success) alert('导出成功！路径: ' + d.path + '\\n共 ' + d.count + ' 条数据');
      else alert('导出失败: ' + d.error);
    });
}

function navigate(step) {
  fetch(`/api/total?filter=${currentFilter}`).then(r=>r.json()).then(d => {
    let next = currentIdx + step;
    if (next < 0) next = 0;
    if (next >= d.total) { alert('已是最后一张'); return; }
    loadData(next);
  });
}

document.addEventListener('keydown', e => {
  if (e.target.tagName==='INPUT' || e.target.tagName==='TEXTAREA' || e.target.tagName==='SELECT') return;
  if (e.key==='ArrowRight') navigate(1);
  else if (e.key==='ArrowLeft') navigate(-1);
  else if (e.key==='s'||e.key==='S') saveAndNext();
});

window.onload = () => loadData(0);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    load_cache()
    return render_template_string(HTML)

@app.route("/api/total")
def api_total():
    return jsonify({"total": len(filter_keys(request.args.get("filter", "all")))})

@app.route("/api/data")
def api_data():
    idx = int(request.args.get("idx", 0))
    f = request.args.get("filter", "all")
    keys = filter_keys(f)
    if not keys:
        return jsonify({"error": "没有符合条件的数据"})
    idx = max(0, min(idx, len(keys) - 1))
    key = keys[idx]
    return jsonify({
        "idx": idx, "total": len(keys),
        "key": key,
        "filename": Path(key).name,
        "ann": copy.deepcopy(cache[key]),
    })

@app.route("/api/image")
def api_image():
    idx = int(request.args.get("idx", 0))
    f = request.args.get("filter", "all")
    keys = filter_keys(f)
    if not keys:
        return "no data", 404
    key = keys[max(0, min(idx, len(keys) - 1))]
    img_path = cache[key]["image"]
    if not Path(img_path).exists():
        return "image not found", 404
    suffix = Path(img_path).suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else f"image/{suffix.lstrip('.')}"
    return send_file(img_path, mimetype=mime)

@app.route("/api/exclude", methods=["POST"])
def api_exclude():
    data = request.json
    keys = filter_keys(data["filter"])
    idx = data["idx"]
    if idx >= len(keys):
        return jsonify({"error": "index out of range"}), 400
    key = keys[idx]
    del cache[key]
    save_cache()
    load_cache()
    return jsonify({"success": True})

@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.json
    keys = filter_keys(data["filter"])
    idx = data["idx"]
    if idx >= len(keys):
        return jsonify({"error": "index out of range"}), 400
    key = keys[idx]
    
    ann = cache[key]
    ann["water_status"] = data["water_status"]
    ann["description"] = data["description"]
    ann["reviewed"] = True
    
    save_cache()
    load_cache()
    return jsonify({"success": True})

@app.route("/api/export", methods=["POST"])
def api_export():
    try:
        # 只导出已审查的数据
        reviewed_keys = [k for k in cache.keys() if cache[k].get("reviewed")]
        samples = []
        
        system_prompt = (
            "你是水环境保护专家，基于无人机俯拍图片判断水体状态是否正常，输出结构化 JSON 结论。\n"
            "【水体状态判定标准】：\n"
            "- water_status=\"normal\"：水体清澈或轻度浑浊，无大面积藻类、漂浮物、泡沫或异常颜色。\n"
            "- water_status=\"polluted\"：水面存在大面积藻类/浮萍、明显泡沫/油污、异常颜色（如发黑/发红/发绿）、大量漂浮垃圾、或水体严重浑浊。\n"
            "【输出要求】：\n"
            "只输出一个 JSON 对象，包含 water_status（\"normal\" 或 \"polluted\"） 和 description（一句话描述原因）。"
        )
        
        for k in reviewed_keys:
            ann = cache[k]
            # 这里构建 Swift / Qwen-VL 所需的多模态格式
            sample = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "<image>请分析这张水域图片，判断水体状态是否正常，以JSON格式输出结果。"},
                    {"role": "assistant", "content": json.dumps({
                        "water_status": ann["water_status"],
                        "description": ann["description"]
                    }, ensure_ascii=False)}
                ],
                "images": [ann["image"]]
            }
            samples.append(sample)
            
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\\n")
                
        return jsonify({"success": True, "path": str(OUTPUT_JSONL), "count": len(samples)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    load_cache()
    total = len(cache)
    unreviewed = sum(1 for a in cache.values() if not a.get("reviewed"))
    print(f"加载 {total} 条标注，未审查 {unreviewed} 条")
    print("访问地址: http://localhost:5002")
    print("SSH 端口转发: ssh -L 5002:localhost:5002 user@server")
    app.run(host="0.0.0.0", port=5002, debug=False)
