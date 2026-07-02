"""
Browser-based label reviewer for MIPA photos.

Shows each image with its auto-labeled bounding boxes overlaid.
You can approve, delete individual boxes, draw new ones, and navigate.
All changes save directly to the YOLO .txt files.

Usage:
    venv/bin/python scripts/review_labels.py
    Then open http://localhost:7777 in your browser.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

REPO_ROOT = Path(__file__).resolve().parent.parent
PHOTO_DIR = REPO_ROOT / "data" / "raw" / "mipa_photos"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

app = Flask(__name__, static_folder=None)

images = sorted(f for f in PHOTO_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def read_labels(img_path: Path) -> list[list[float]]:
    txt = img_path.with_suffix(".txt")
    if not txt.exists():
        return []
    boxes = []
    for line in txt.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append([float(x) for x in parts[1:]])
    return boxes


def write_labels(img_path: Path, boxes: list[list[float]]) -> None:
    txt = img_path.with_suffix(".txt")
    lines = [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in boxes]
    txt.write_text("\n".join(lines))


def encode_image(img_path: Path, boxes: list[list[float]]) -> str:
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    for box in boxes:
        xc, yc, bw, bh = box
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # Resize for display (max 900px wide)
    if w > 900:
        scale = 900 / w
        img = cv2.resize(img, (900, int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/image/<int:idx>")
def get_image(idx: int):
    if idx < 0 or idx >= len(images):
        return jsonify(error="out of range"), 404
    img_path = images[idx]
    boxes = read_labels(img_path)
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2] if img is not None else (1, 1)
    return jsonify(
        idx=idx,
        total=len(images),
        name=img_path.name,
        has_label=img_path.with_suffix(".txt").exists(),
        boxes=boxes,
        orig_w=w,
        orig_h=h,
        image_b64=encode_image(img_path, boxes),
    )


@app.route("/api/save/<int:idx>", methods=["POST"])
def save(idx: int):
    if idx < 0 or idx >= len(images):
        return jsonify(error="out of range"), 404
    data = request.get_json()
    boxes = data.get("boxes", [])
    write_labels(images[idx], boxes)
    return jsonify(ok=True)


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Label Reviewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a1a; color: #eee; font-family: monospace; }
#top { display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: #111; border-bottom: 1px solid #333; }
#top button { background: #333; color: #eee; border: 1px solid #555; padding: 6px 14px; cursor: pointer; border-radius: 4px; font-size: 14px; }
#top button:hover { background: #444; }
#top button.green { background: #1a5c1a; border-color: #2a8c2a; }
#top button.red { background: #5c1a1a; border-color: #8c2a2a; }
#info { flex: 1; font-size: 13px; color: #aaa; }
#status { font-size: 13px; color: #7fc97f; min-width: 200px; text-align: right; }
#canvas-wrap { position: relative; display: inline-block; margin: 20px auto; display: block; text-align: center; }
canvas { cursor: crosshair; border: 2px solid #333; }
#hint { text-align: center; font-size: 12px; color: #666; margin-top: 8px; }
#progress { background: #333; height: 4px; }
#progress-bar { background: #2a8c2a; height: 4px; transition: width 0.2s; }
</style>
</head>
<body>
<div id="progress"><div id="progress-bar" style="width:0%"></div></div>
<div id="top">
  <button onclick="nav(-1)">&#9664; Prev (A)</button>
  <button onclick="nav(1)">Next (D) &#9654;</button>
  <button class="red" onclick="clearBoxes()">Clear All (C)</button>
  <button class="green" onclick="save()">Save &amp; Next (S)</button>
  <div id="info">Loading...</div>
  <div id="status"></div>
</div>
<div id="canvas-wrap"><canvas id="c"></canvas></div>
<div id="hint">Click &amp; drag to draw a box &nbsp;|&nbsp; Right-click a box to delete it &nbsp;|&nbsp; S = save &amp; next &nbsp;|&nbsp; A/D = prev/next</div>

<script>
let idx = 0, total = 0, boxes = [], drawing = false, startX, startY, origW, origH, scale = 1;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let img = new Image();

function load(i) {
  fetch('/api/image/' + i)
    .then(r => r.json())
    .then(d => {
      idx = d.idx; total = d.total; boxes = d.boxes; origW = d.orig_w; origH = d.orig_h;
      document.getElementById('info').textContent = (idx+1) + ' / ' + total + '  —  ' + d.name + (d.has_label ? '' : '  [NO LABEL]');
      document.getElementById('progress-bar').style.width = ((idx+1)/total*100) + '%';
      img.src = 'data:image/jpeg;base64,' + d.image_b64;
      img.onload = redraw;
      setStatus('loaded');
    });
}

function redraw() {
  const maxW = Math.min(900, window.innerWidth - 40);
  scale = maxW / origW;
  canvas.width = Math.round(origW * scale);
  canvas.height = Math.round(origH * scale);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  boxes.forEach((b, i) => drawBox(b, i));
}

function drawBox(b, i) {
  const [xc, yc, bw, bh] = b;
  const x = (xc - bw/2) * origW * scale;
  const y = (yc - bh/2) * origH * scale;
  const w = bw * origW * scale;
  const h = bh * origH * scale;
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = 'rgba(0,255,0,0.15)';
  ctx.fillRect(x, y, w, h);
}

canvas.addEventListener('mousedown', e => {
  if (e.button === 2) { rightClick(e); return; }
  drawing = true;
  const r = canvas.getBoundingClientRect();
  startX = e.clientX - r.left;
  startY = e.clientY - r.top;
});

canvas.addEventListener('mouseup', e => {
  if (!drawing) return;
  drawing = false;
  const r = canvas.getBoundingClientRect();
  const ex = e.clientX - r.left, ey = e.clientY - r.top;
  const x1 = Math.min(startX, ex), y1 = Math.min(startY, ey);
  const x2 = Math.max(startX, ex), y2 = Math.max(startY, ey);
  if (x2 - x1 < 10 || y2 - y1 < 10) return;
  const xc = ((x1 + x2) / 2) / (origW * scale);
  const yc = ((y1 + y2) / 2) / (origH * scale);
  const bw = (x2 - x1) / (origW * scale);
  const bh = (y2 - y1) / (origH * scale);
  boxes.push([xc, yc, bw, bh]);
  redraw();
});

canvas.addEventListener('contextmenu', e => { e.preventDefault(); rightClick(e); });

function rightClick(e) {
  const r = canvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  for (let i = boxes.length - 1; i >= 0; i--) {
    const [xc, yc, bw, bh] = boxes[i];
    const x = (xc - bw/2) * origW * scale, y = (yc - bh/2) * origH * scale;
    const w = bw * origW * scale, h = bh * origH * scale;
    if (mx >= x && mx <= x+w && my >= y && my <= y+h) {
      boxes.splice(i, 1); redraw(); setStatus('deleted box'); return;
    }
  }
}

function clearBoxes() { boxes = []; redraw(); setStatus('cleared'); }

function save() {
  fetch('/api/save/' + idx, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({boxes}) })
    .then(() => { setStatus('saved ✓'); nav(1); });
}

function nav(d) { const ni = idx + d; if (ni >= 0 && ni < total) load(ni); }

function setStatus(msg) { document.getElementById('status').textContent = msg; }

document.addEventListener('keydown', e => {
  if (e.key === 'a' || e.key === 'A') nav(-1);
  if (e.key === 'd' || e.key === 'D') nav(1);
  if (e.key === 's' || e.key === 'S') save();
  if (e.key === 'c' || e.key === 'C') clearBoxes();
});

load(0);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"[load] {len(images)} images in {PHOTO_DIR.relative_to(REPO_ROOT)}")
    print(f"[open] http://localhost:7777")
    app.run(port=7777, debug=False)
