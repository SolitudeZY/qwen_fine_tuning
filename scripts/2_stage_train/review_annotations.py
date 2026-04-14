"""
canonical annotation 人工审查 Web UI
读取 data/annotate_cache.json，支持：
  - Canvas 前端叠加 bbox（无缩放错位）
  - 鼠标拖拽在图上新增违规框
  - 修改 label / 删除单个框 / 整体标记合规或违规
  - 审查结果写回 annotate_cache.json
  - 低置信度图片优先排序
  - 过滤：全部 / 未审查 / 低置信度 / 已审查

用法：
  conda activate qwen-ft
  python scripts/2_stage_train/review_annotations.py
  # 本机 SSH 端口转发：ssh -L 5001:localhost:5001 user@server
  # 浏览器打开 http://localhost:5001
"""

import copy
import json
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file

ROOT = Path(__file__).parent.parent.parent
CACHE_FILE = ROOT / "data" / "annotate_cache.json"

VALID_LABELS = ["围栏断口", "围栏倒伏", "临边防护缺失", "临边开口未防护"]
VALID_SCENES = ["基坑", "边坡", "泥浆池", "平台临边", "其他"]
LABEL_COLORS = {
    "围栏断口":       "#FF4444",
    "围栏倒伏":       "#FF8800",
    "临边防护缺失":   "#CC00FF",
    "临边开口未防护": "#00AAFF",
}

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
    conf_order = {"low": 0, "medium": 1, "high": 2}
    ordered_keys = sorted(
        cache.keys(),
        key=lambda k: (
            1 if cache[k].get("reviewed") else 0,
            conf_order.get(cache[k].get("confidence", "medium"), 1),
        ),
    )


def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def filter_keys(mode: str) -> list[str]:
    if mode == "unreviewed":
        return [k for k in ordered_keys if not cache[k].get("reviewed")]
    if mode == "low_conf":
        return [k for k in ordered_keys if cache[k].get("confidence") == "low"]
    if mode == "reviewed":
        return [k for k in ordered_keys if cache[k].get("reviewed")]
    return list(ordered_keys)


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>标注审查工具</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Microsoft YaHei", Arial, sans-serif; background: #1a1a2e; color: #eee; }
.layout { display: flex; height: 100vh; }
.sidebar { width: 320px; min-width: 320px; background: #16213e; padding: 16px; overflow-y: auto; border-right: 1px solid #0f3460; flex-shrink: 0; }
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
.toolbar { background: #0f3460; padding: 10px 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; flex-shrink: 0; }
.image-area { flex: 1; overflow: auto; display: flex; justify-content: center; align-items: flex-start; padding: 16px; background: #0d0d1a; position: relative; }
#canvasWrap { position: relative; display: inline-block; line-height: 0; }
#bgImg { display: block; max-width: 100%; max-height: calc(100vh - 120px); }
#drawCanvas { position: absolute; top: 0; left: 0; cursor: crosshair; }
h2 { color: #e94560; margin-bottom: 12px; font-size: 16px; }
h3 { color: #a8b2d8; font-size: 13px; margin: 12px 0 6px; }
.meta-row { font-size: 12px; color: #a8b2d8; margin: 4px 0; }
.meta-row span { color: #eee; font-weight: bold; }
.conf-high { color: #4caf50; } .conf-medium { color: #ff9800; } .conf-low { color: #f44336; font-weight: bold; }
.violation-card { background: #0f3460; border-radius: 6px; padding: 10px; margin: 6px 0; border-left: 4px solid #e94560; cursor: pointer; }
.violation-card:hover { background: #1a3a6e; }
.violation-card.deleted { opacity: 0.4; }
.violation-card.active-card { outline: 2px solid #e94560; }
.v-evidence { font-size: 11px; color: #a8b2d8; margin: 4px 0; }
select, input[type=text] { background: #1a1a2e; color: #eee; border: 1px solid #0f3460; border-radius: 4px; padding: 4px 8px; font-size: 12px; }
select:focus, input:focus { outline: none; border-color: #e94560; }
button { cursor: pointer; border: none; border-radius: 4px; padding: 6px 14px; font-size: 13px; font-weight: bold; transition: opacity .2s; }
button:hover { opacity: 0.85; }
.btn-success { background: #4caf50; color: white; } .btn-danger { background: #f44336; color: white; }
.btn-info { background: #2196f3; color: white; } .btn-warning { background: #ff9800; color: white; }
.btn-sm { padding: 3px 8px; font-size: 11px; }
.progress { font-size: 13px; color: #a8b2d8; }
.filter-bar { display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }
.filter-btn { background: #0f3460; color: #a8b2d8; padding: 4px 10px; font-size: 12px; border-radius: 12px; border: 1px solid #0f3460; }
.filter-btn.active { background: #e94560; color: white; border-color: #e94560; }
.divider { border: none; border-top: 1px solid #0f3460; margin: 10px 0; }
.kbd { background: #0f3460; border-radius: 3px; padding: 1px 5px; font-size: 11px; font-family: monospace; }
.draw-hint { font-size: 11px; color: #ff9800; margin-bottom: 6px; }
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <h2>标注审查工具</h2>
    <div class="filter-bar">
      <button class="filter-btn active" onclick="setFilter('all',this)">全部</button>
      <button class="filter-btn" onclick="setFilter('unreviewed',this)">未审查</button>
      <button class="filter-btn" onclick="setFilter('low_conf',this)">低置信度</button>
      <button class="filter-btn" onclick="setFilter('reviewed',this)">已审查</button>
    </div>
    <div class="progress" id="progress">加载中...</div>
    <hr class="divider">
    <div class="meta-row">文件名：<span id="mFilename">-</span></div>
    <div class="meta-row">场景：
      <select id="editScene" style="margin-left:4px"></select>
    </div>
    <div class="meta-row">置信度：<span id="mConf">-</span></div>
    <div class="meta-row">审查状态：<span id="mReviewed">-</span></div>
    <div style="margin-top:6px">
      <div class="meta-row" style="margin-bottom:4px">整改建议</div>
      <input type="text" id="editSuggestion" style="width:100%" placeholder="整改建议">
    </div>
    <hr class="divider">

    <h3>违规框
      <select id="newLabelSelect" style="margin-left:6px;font-size:11px"></select>
      <span style="font-size:11px;color:#a8b2d8;margin-left:4px">在图上拖拽新增</span>
    </h3>
    <div class="draw-hint" id="drawHint">在图片上按住鼠标拖拽即可框选新违规区域</div>
    <div id="violationList"></div>
    <hr class="divider">

    <div style="display:flex;flex-direction:column;gap:8px;">
      <button class="btn-success" onclick="saveAndNext(false)">✅ 合规并保存 <span class="kbd">N</span></button>
      <button class="btn-danger"  onclick="saveAndNext(true)">🚨 违规并保存 <span class="kbd">Y</span></button>
      <button class="btn-info"    onclick="saveOnly()">💾 仅保存 <span class="kbd">S</span></button>
      <button class="btn-warning" onclick="excludeImage()" style="margin-top:4px">🗑 排除此图（无关场景）</button>
    </div>
    <hr class="divider">
    <div style="font-size:11px;color:#666;line-height:2">
      <span class="kbd">←</span><span class="kbd">→</span> 翻页 &nbsp;
      <span class="kbd">Y</span> 违规 &nbsp;
      <span class="kbd">N</span> 合规 &nbsp;
      <span class="kbd">S</span> 保存
    </div>
  </div>

  <div class="main">
    <div class="toolbar">
      <button class="btn-info" onclick="navigate(-1)">← 上一张</button>
      <button class="btn-info" onclick="navigate(1)">下一张 →</button>
      <span id="toolbarInfo" style="font-size:13px;color:#a8b2d8"></span>
    </div>
    <div class="image-area">
      <div id="canvasWrap">
        <img id="bgImg" src="" alt="加载中..." onload="onImageLoad()">
        <canvas id="drawCanvas"></canvas>
      </div>
    </div>
  </div>
</div>

<script>
const VALID_LABELS = {{ valid_labels|tojson }};
const VALID_SCENES = {{ valid_scenes|tojson }};
const LABEL_COLORS = {{ label_colors|tojson }};

let currentIdx = 0, currentFilter = 'all', ann = null;
let imgNaturalW = 1, imgNaturalH = 1;
let isDragging = false, dragStart = null, dragRect = null;

// 初始化 select
const sceneEl = document.getElementById('editScene');
VALID_SCENES.forEach(s => { const o = document.createElement('option'); o.value=s; o.textContent=s; sceneEl.appendChild(o); });
const newLabelEl = document.getElementById('newLabelSelect');
VALID_LABELS.forEach(l => { const o = document.createElement('option'); o.value=l; o.textContent=l; newLabelEl.appendChild(o); });

// bbox 归一化：clamp + 坐标顺序修正
function normBbox(b) {
  if (!b || b.length !== 4) return [0, 0, 100, 100];
  let [x0, y0, x1, y1] = b.map(v => Math.max(0, Math.min(1000, v)));
  if (x0 > x1) [x0, x1] = [x1, x0];
  if (y0 > y1) [y0, y1] = [y1, y0];
  return [x0, y0, x1, y1];
}

function isBboxDirty(b) {
  if (!b || b.length !== 4) return true;
  return b.some(v => v < 0 || v > 1000) || b[0] > b[2] || b[1] > b[3];
}

// ── Canvas 绘制 ──────────────────────────────────────────────────────────────
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const bgImg = document.getElementById('bgImg');

function onImageLoad() {
  imgNaturalW = bgImg.naturalWidth;
  imgNaturalH = bgImg.naturalHeight;
  resizeCanvas();
  redraw();
}

function resizeCanvas() {
  canvas.width  = bgImg.offsetWidth;
  canvas.height = bgImg.offsetHeight;
  canvas.style.width  = bgImg.offsetWidth  + 'px';
  canvas.style.height = bgImg.offsetHeight + 'px';
}

window.addEventListener('resize', () => { resizeCanvas(); redraw(); });

// 0-1000 → canvas 像素
function toCanvas(bx, by) {
  return [bx / 1000 * canvas.width, by / 1000 * canvas.height];
}
// canvas 像素 → 0-1000
function toNorm(cx, cy) {
  return [Math.round(cx / canvas.width * 1000), Math.round(cy / canvas.height * 1000)];
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!ann) return;
  (ann.violations || []).forEach((v, i) => {
    if (v._deleted) return;
    const dirty = isBboxDirty(v.bbox_1000);
    const b = normBbox(v.bbox_1000);
    const [x0, y0] = toCanvas(b[0], b[1]);
    const [x1, y1] = toCanvas(b[2], b[3]);
    const color = LABEL_COLORS[v.label] || '#FFFF00';
    ctx.strokeStyle = dirty ? '#FF0000' : color;
    ctx.lineWidth = dirty ? 3 : 2;
    if (dirty) ctx.setLineDash([6, 3]);
    ctx.strokeRect(x0, y0, x1-x0, y1-y0);
    ctx.setLineDash([]);
    ctx.font = 'bold 13px Arial';
    const tw = ctx.measureText(v.label).width;
    ctx.fillStyle = dirty ? '#FF0000' : color;
    ctx.fillRect(x0, y0 - 18, tw + 8, 18);
    ctx.fillStyle = '#fff';
    ctx.fillText(v.label, x0 + 4, y0 - 4);
  });
  // 拖拽中的新框
  if (isDragging && dragRect) {
    ctx.strokeStyle = '#FFFF00';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(dragRect.x, dragRect.y, dragRect.w, dragRect.h);
    ctx.setLineDash([]);
  }
}

// ── 鼠标拖拽新增框 ────────────────────────────────────────────────────────────
function getCanvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  isDragging = true;
  dragStart = getCanvasPos(e);
  dragRect = { x: dragStart.x, y: dragStart.y, w: 0, h: 0 };
});

canvas.addEventListener('mousemove', e => {
  if (!isDragging) return;
  const pos = getCanvasPos(e);
  dragRect = {
    x: Math.min(dragStart.x, pos.x),
    y: Math.min(dragStart.y, pos.y),
    w: Math.abs(pos.x - dragStart.x),
    h: Math.abs(pos.y - dragStart.y),
  };
  redraw();
});

canvas.addEventListener('mouseup', e => {
  if (!isDragging) return;
  isDragging = false;
  if (!dragRect || dragRect.w < 5 || dragRect.h < 5) { dragRect = null; redraw(); return; }
  const [x0, y0] = toNorm(dragRect.x, dragRect.y);
  const [x1, y1] = toNorm(dragRect.x + dragRect.w, dragRect.y + dragRect.h);
  const label = newLabelEl.value;
  ann.violations = ann.violations || [];
  ann.violations.push({ label, bbox_1000: [x0, y0, x1, y1], severity: '中', evidence: '手动标注' });
  dragRect = null;
  renderViolations();
  redraw();
});

canvas.addEventListener('mouseleave', e => {
  if (isDragging) { isDragging = false; dragRect = null; redraw(); }
});

// ── 数据加载 ──────────────────────────────────────────────────────────────────
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
      document.getElementById('toolbarInfo').textContent = `${data.filename}  [${ann.scene}]  confidence: ${ann.confidence}`;
      document.getElementById('mFilename').textContent = data.filename;
      const confEl = document.getElementById('mConf');
      confEl.textContent = ann.confidence; confEl.className = 'conf-' + ann.confidence;
      document.getElementById('mReviewed').textContent = ann.reviewed ? '✅ 已审查' : '⚪ 未审查';
      sceneEl.value = ann.scene;
      document.getElementById('editSuggestion').value = ann.global_conclusion?.suggestion || '';
      renderViolations();
      // 加载原图
      bgImg.src = `/api/image?idx=${data.idx}&filter=${currentFilter}&t=${Date.now()}`;
    });
}

function renderViolations() {
  const list = document.getElementById('violationList');
  list.innerHTML = '';
  (ann.violations || []).forEach((v, i) => {
    const color = LABEL_COLORS[v.label] || '#aaa';
    const card = document.createElement('div');
    card.className = 'violation-card' + (v._deleted ? ' deleted' : '');
    card.style.borderLeftColor = color;
    card.innerHTML = `
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
        <select onchange="updateLabel(${i},this.value)"
          style="background:#1a1a2e;color:${color};border:1px solid ${color};border-radius:3px;font-weight:bold;font-size:12px;padding:2px 4px">
          ${VALID_LABELS.map(l=>`<option value="${l}"${l===v.label?' selected':''}>${l}</option>`).join('')}
        </select>
        <select onchange="updateSeverity(${i},this.value)"
          style="background:#1a1a2e;color:#a8b2d8;border:1px solid #0f3460;border-radius:3px;font-size:11px;padding:2px 4px">
          ${['低','中','高'].map(s=>`<option value="${s}"${s===v.severity?' selected':''}>${s}</option>`).join('')}
        </select>
      </div>
      <div class="v-evidence" style="font-size:11px;color:#a8b2d8">
        ${isBboxDirty(v.bbox_1000) ? '<span style="color:#f44336;font-weight:bold">⚠ 坐标越界，保存时自动修正</span><br>' : ''}
        bbox: [${(v.bbox_1000||[]).map(x=>Math.round(x)).join(', ')}]
      </div>
      <div style="margin-top:4px">
        ${v._deleted
          ? `<button class="btn-sm btn-success" onclick="restoreViolation(${i})">恢复</button>`
          : `<button class="btn-sm btn-danger" onclick="deleteViolation(${i})">删除</button>`}
      </div>`;
    list.appendChild(card);
  });
}

function updateLabel(i, val) { ann.violations[i].label = val; renderViolations(); redraw(); }
function updateSeverity(i, val) { ann.violations[i].severity = val; }
function deleteViolation(i) { ann.violations[i]._deleted = true; renderViolations(); redraw(); }
function restoreViolation(i) { delete ann.violations[i]._deleted; renderViolations(); redraw(); }

function buildPayload(isCompliant) {
  const violations = (ann.violations || [])
    .filter(v => !v._deleted)
    .map(v => ({ ...v, bbox_1000: normBbox(v.bbox_1000) }));
  return {
    idx: currentIdx, filter: currentFilter,
    scene: sceneEl.value,
    suggestion: document.getElementById('editSuggestion').value,
    violations, is_compliant: isCompliant,
    violation_detected: !isCompliant && violations.length > 0,
  };
}

function saveAndNext(isViolation) {
  fetch('/api/save', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(buildPayload(!isViolation))
  }).then(() => navigate(1));
}

function saveOnly() {
  fetch('/api/save', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(buildPayload(ann.is_compliant))
  }).then(r=>r.json()).then(() => loadData(currentIdx));
}

function excludeImage() {
  if (!confirm('确认排除此图？它将从 cache 中移除，不会出现在训练数据里。')) return;
  fetch('/api/exclude', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ idx: currentIdx, filter: currentFilter })
  }).then(r=>r.json()).then(d => {
    if (d.error) { alert(d.error); return; }
    loadData(currentIdx);  // 排除后当前位置自动显示下一张
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
  if (e.target.tagName==='INPUT' || e.target.tagName==='SELECT') return;
  if (e.key==='ArrowRight') navigate(1);
  else if (e.key==='ArrowLeft') navigate(-1);
  else if (e.key==='y'||e.key==='Y') saveAndNext(true);
  else if (e.key==='n'||e.key==='N') saveAndNext(false);
  else if (e.key==='s'||e.key==='S') saveOnly();
});

window.onload = () => loadData(0);
</script>
</body>
</html>
"""

app = Flask(__name__)


@app.route("/")
def index():
    load_cache()
    return render_template_string(
        HTML,
        valid_labels=VALID_LABELS,
        valid_scenes=VALID_SCENES,
        label_colors=LABEL_COLORS,
    )


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
    # 从 cache 移除
    del cache[key]
    save_cache()
    # 写入 skip 列表
    skip_file = ROOT / "data" / "annotate_skip.json"
    skip = json.loads(skip_file.read_text(encoding="utf-8")) if skip_file.exists() else []
    skip.append({"image": key, "reason": "excluded_by_reviewer: 无关场景"})
    skip_file.write_text(json.dumps(skip, ensure_ascii=False, indent=2), encoding="utf-8")
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
    violations = data["violations"]
    labels = list(dict.fromkeys(v["label"] for v in violations))
    ann["scene"] = data["scene"]
    ann["is_compliant"] = data["is_compliant"]
    ann["violations"] = violations
    ann["reviewed"] = True
    ann["global_conclusion"]["violation_detected"] = data["violation_detected"]
    ann["global_conclusion"]["violation_type"] = "、".join(labels) if labels else ""
    ann["global_conclusion"]["suggestion"] = data["suggestion"]
    if data["is_compliant"]:
        ann["global_conclusion"]["severity"] = "无"
    save_cache()
    load_cache()
    return jsonify({"success": True})


if __name__ == "__main__":
    load_cache()
    total = len(cache)
    low_conf = sum(1 for a in cache.values() if a.get("confidence") == "low")
    unreviewed = sum(1 for a in cache.values() if not a.get("reviewed"))
    print(f"加载 {total} 条标注，低置信度 {low_conf} 条，未审查 {unreviewed} 条")
    print("访问地址: http://localhost:5001")
    print("SSH 端口转发: ssh -L 5001:localhost:5001 user@server")
    app.run(host="0.0.0.0", port=5001, debug=False)
