import argparse
import colorsys
import json
import random
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from trocr_handwritten.parse.settings import CLASS_NAMES, CLASS_NAMES_LIST
from trocr_handwritten.parse.utils import _load_model
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

LAYOUT_DIR = Path("data/layout")
SPLIT_WEIGHTS = {"train": 0.7, "test": 0.2, "dev": 0.1}
SPLITS = list(SPLIT_WEIGHTS.keys())


def _generate_colors(n):
    """
    Generate n visually distinct colors using HSV spacing.

    Args:
        n: Number of colors to generate

    Returns:
        list: List of hex color strings
    """
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def _assign_split():
    """Randomly assign a split based on SPLIT_WEIGHTS (train=0.7, test=0.2, dev=0.1)."""
    r = random.random()
    cumulative = 0.0
    for split, weight in SPLIT_WEIGHTS.items():
        cumulative += weight
        if r < cumulative:
            return split
    return "train"


def _load_annotations():
    """Load existing annotations from all split JSON files into a single list."""
    all_annotations = []
    for split in SPLITS:
        path = LAYOUT_DIR / split / "annotations.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                entries = json.load(f)
                for entry in entries:
                    entry["split"] = split
                all_annotations.extend(entries)
    return all_annotations


def _save_annotations(annotations):
    """Save annotations to per-split JSON files."""
    by_split = {s: [] for s in SPLITS}
    for a in annotations:
        split = a.get("split", "train")
        by_split[split].append(a)

    for split, entries in by_split.items():
        split_dir = LAYOUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / "annotations.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)


def _collect_images(images_dir):
    """Collect all image files from a directory."""
    extensions = ("*.jpg", "*.jpeg", "*.png")
    images = []
    for ext in extensions:
        images.extend(Path(images_dir).glob(ext))
    return sorted(images)


def _prefill_image(image_path, model):
    """Run YOLO model on an image and return detected boxes."""
    results = model.predict(
        str(image_path), imgsz=1024, conf=0.15, iou=0.5, verbose=False
    )
    boxes = []
    if results and len(results) > 0:
        det = results[0]
        if det.boxes is not None and len(det.boxes) > 0:
            xyxy = det.boxes.xyxy.cpu().numpy()
            classes = det.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                class_id = int(classes[i])
                label = CLASS_NAMES.get(str(class_id))
                if label is None:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                w, h = float(x2 - x1), float(y2 - y1)
                if w < 15 or h < 15:
                    continue
                boxes.append(
                    {
                        "class_id": class_id,
                        "label": label,
                        "x": float(x1),
                        "y": float(y1),
                        "width": w,
                        "height": h,
                    }
                )
    return boxes


ANNOTATE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1a1a2e; color: #eee; height: 100vh; display: flex; flex-direction: column;
       overflow: hidden; user-select: none; }
.header { background: #16213e; padding: 0.6rem 1.5rem; display: flex;
          align-items: center; justify-content: space-between; flex-shrink: 0; }
.header h1 { font-size: 0.95rem; font-weight: 500; color: #a8b2d1; }
.header-right { display: flex; align-items: center; gap: 1rem; }
.progress { font-size: 0.8rem; color: #8892b0; }
.stats { font-size: 0.75rem; color: #64ffda; }
.main { flex: 1; display: flex; flex-direction: column; align-items: center;
        overflow: hidden; padding: 0.5rem; min-height: 0; }
.info { font-size: 0.8rem; color: #8892b0; margin-bottom: 0.3rem; text-align: center; }
.info strong { color: #ccd6f6; }
.canvas-wrap { flex: 1; display: flex; justify-content: center; align-items: center;
               min-height: 0; width: 100%; position: relative; overflow: auto; }
.canvas-container { position: relative; display: inline-block;
                    transform-origin: center center; transition: transform 0.1s ease; }
.canvas-container img { display: block; max-width: 100%; max-height: calc(100vh - 180px);
                        object-fit: contain; }
.canvas-container canvas { position: absolute; top: 0; left: 0; cursor: crosshair; }
.zoom-info { position: absolute; bottom: 0.5rem; right: 0.5rem; font-size: 0.7rem;
             color: #8892b0; background: rgba(15,23,41,0.8); padding: 0.2rem 0.5rem;
             border-radius: 3px; pointer-events: none; z-index: 10; }
.toolbar { background: #16213e; padding: 0.5rem 1rem; display: flex;
           justify-content: center; align-items: center; gap: 0.8rem; flex-shrink: 0; }
.label-btn { padding: 0.4rem 1rem; border: 2px solid transparent; border-radius: 5px;
             font-size: 0.85rem; font-weight: 600; cursor: pointer; background: #233554;
             color: #ccd6f6; transition: all 0.1s; }
.label-btn.active { border-color: currentColor; }
.label-btn .key { font-size: 0.65rem; color: #8892b0; margin-left: 0.3rem; }
.controls { background: #0f1729; padding: 0.5rem 1rem; display: flex;
            justify-content: center; gap: 0.6rem; flex-shrink: 0; }
.btn { padding: 0.5rem 1.5rem; border: none; border-radius: 5px; font-size: 0.85rem;
       font-weight: 600; cursor: pointer; transition: all 0.1s; }
.btn-nav { background: #495670; color: #ccd6f6; }
.btn-nav:hover { background: #5a6a8a; }
.btn-save { background: #64ffda; color: #0a192f; }
.btn-save:hover { background: #4cd9b0; }
.btn-prefill { background: #c792ea; color: #0a192f; }
.btn-prefill:hover { background: #b07cd8; }
.btn-clear { background: #f07178; color: #0a192f; }
.btn-clear:hover { background: #d45d63; }
.shortcut { font-size: 0.65rem; color: #8892b0; display: block; margin-top: 0.1rem; }
.toast { position: fixed; top: 1rem; right: 1rem; padding: 0.6rem 1.2rem;
         background: #64ffda; color: #0a192f; border-radius: 5px; font-weight: 600;
         font-size: 0.85rem; opacity: 0; transition: opacity 0.3s; z-index: 999;
         pointer-events: none; }
.toast.show { opacity: 1; }
"""

ANNOTATE_JS = """
const CLASSES = INJECT_CLASSES;
const COLORS = INJECT_COLORS;

let boxes = INITIAL_BOXES;
let selectedIdx = -1;
let currentLabel = 0;
let scale = 1;
let imgW = 0, imgH = 0;
let drag = null;
let zoom = 1;

const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const img = document.getElementById('page-img');
const toast = document.getElementById('toast');
const container = document.querySelector('.canvas-container');
const zoomInfo = document.getElementById('zoom-info');

function applyTransform() {
    container.style.transform = 'scale(' + zoom + ')';
    if (zoomInfo) zoomInfo.textContent = Math.round(zoom * 100) + '%';
}

function initCanvas() {
    const rect = img.getBoundingClientRect();
    const baseW = img.naturalWidth;
    const baseH = img.naturalHeight;
    const displayW = rect.width / zoom;
    const displayH = rect.height / zoom;
    canvas.width = displayW;
    canvas.height = displayH;
    scale = displayW / baseW;
    imgW = baseW;
    imgH = baseH;
    render();
}

img.addEventListener('load', () => { initCanvas(); applyTransform(); });
window.addEventListener('resize', initCanvas);

function toImg(cx, cy) { return [cx / scale, cy / scale]; }
function toCvs(ix, iy) { return [ix * scale, iy * scale]; }

function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    boxes.forEach((box, i) => {
        const [x, y] = toCvs(box.x, box.y);
        const w = box.width * scale, h = box.height * scale;
        const color = COLORS[box.class_id] || '#fff';

        ctx.strokeStyle = color;
        ctx.lineWidth = i === selectedIdx ? 3 : 2;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = color;
        ctx.globalAlpha = 0.08;
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1;

        ctx.font = 'bold 11px sans-serif';
        const txt = CLASSES[box.class_id] || '?';
        const tw = ctx.measureText(txt).width + 8;
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 16, tw, 16);
        ctx.fillStyle = '#0a192f';
        ctx.fillText(txt, x + 4, y - 4);

        if (i === selectedIdx) {
            const hs = 8;
            ctx.fillStyle = color;
            [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([hx,hy]) => {
                ctx.fillRect(hx-hs/2, hy-hs/2, hs, hs);
            });
        }
    });

    if (drag && drag.type === 'draw' && drag.box) {
        const [x, y] = toCvs(drag.box.x, drag.box.y);
        const w = drag.box.width * scale, h = drag.box.height * scale;
        ctx.strokeStyle = COLORS[currentLabel];
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
    }

    updateStats();
}

function updateStats() {
    const el = document.getElementById('box-stats');
    if (el) {
        const counts = Object.keys(CLASSES).map(k => {
            const n = boxes.filter(b => b.class_id === parseInt(k)).length;
            return n > 0 ? CLASSES[k] + '=' + n : '';
        }).filter(Boolean).join(' ');
        el.textContent = counts || 'no boxes';
    }
}

function updateLabelButtons() {
    document.querySelectorAll('.label-btn').forEach((btn, i) => {
        btn.classList.toggle('active', i === currentLabel);
    });
}

function hitHandle(box, mx, my) {
    const hs = 12;
    const [x, y] = toCvs(box.x, box.y);
    const w = box.width * scale, h = box.height * scale;
    const corners = [
        {p:[x,y], c:'nw'}, {p:[x+w,y], c:'ne'},
        {p:[x,y+h], c:'sw'}, {p:[x+w,y+h], c:'se'},
    ];
    for (const corner of corners) {
        if (Math.abs(mx - corner.p[0]) < hs && Math.abs(my - corner.p[1]) < hs)
            return corner.c;
    }
    return null;
}

function hitBox(mx, my) {
    const [ix, iy] = toImg(mx, my);
    for (let i = boxes.length - 1; i >= 0; i--) {
        const b = boxes[i];
        if (ix >= b.x && ix <= b.x + b.width && iy >= b.y && iy <= b.y + b.height)
            return i;
    }
    return -1;
}

canvas.addEventListener('mousedown', (e) => {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;

    if (selectedIdx >= 0) {
        const h = hitHandle(boxes[selectedIdx], mx, my);
        if (h) {
            drag = { type:'resize', handle:h, orig:{...boxes[selectedIdx]} };
            return;
        }
    }

    const hit = hitBox(mx, my);
    if (hit >= 0) {
        selectedIdx = hit;
        const [ix, iy] = toImg(mx, my);
        drag = { type:'move', sx:ix, sy:iy, orig:{...boxes[hit]} };
        render();
        return;
    }

    selectedIdx = -1;
    const [ix, iy] = toImg(mx, my);
    drag = { type:'draw', box:{x:ix, y:iy, width:0, height:0, class_id:currentLabel, label:CLASSES[currentLabel]} };
    render();
});

canvas.addEventListener('mousemove', (e) => {
    if (!drag) return;
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const [ix, iy] = toImg(mx, my);

    if (drag.type === 'draw') {
        drag.box.width = ix - drag.box.x;
        drag.box.height = iy - drag.box.y;
    } else if (drag.type === 'move') {
        const dx = ix - drag.sx, dy = iy - drag.sy;
        boxes[selectedIdx].x = drag.orig.x + dx;
        boxes[selectedIdx].y = drag.orig.y + dy;
    } else if (drag.type === 'resize') {
        const b = boxes[selectedIdx], o = drag.orig;
        if (drag.handle.includes('e')) { b.width = Math.max(10, ix - o.x); }
        if (drag.handle.includes('w')) { b.x = Math.min(ix, o.x+o.width-10); b.width = o.x+o.width - b.x; }
        if (drag.handle.includes('s')) { b.height = Math.max(10, iy - o.y); }
        if (drag.handle.includes('n')) { b.y = Math.min(iy, o.y+o.height-10); b.height = o.y+o.height - b.y; }
    }
    render();
});

canvas.addEventListener('mouseup', () => {
    if (!drag) return;
    if (drag.type === 'draw') {
        let {x, y, width, height, class_id, label} = drag.box;
        if (width < 0) { x += width; width = -width; }
        if (height < 0) { y += height; height = -height; }
        if (width > 5 && height > 5) {
            boxes.push({x, y, width, height, class_id, label});
            selectedIdx = boxes.length - 1;
        }
    }
    drag = null;
    render();
});

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;

    const labelKey = parseInt(e.key);
    if (labelKey >= 1 && labelKey <= Object.keys(CLASSES).length) {
        currentLabel = labelKey - 1;
        if (selectedIdx >= 0) {
            boxes[selectedIdx].class_id = currentLabel;
            boxes[selectedIdx].label = CLASSES[currentLabel];
        }
        updateLabelButtons();
        render();
        return;
    }

    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedIdx >= 0) {
        e.preventDefault();
        boxes.splice(selectedIdx, 1);
        selectedIdx = -1;
        render();
        return;
    }

    if (e.key === 'Escape') { selectedIdx = -1; render(); return; }

    if (e.key === 's' || ((e.ctrlKey || e.metaKey) && e.key === 's')) {
        e.preventDefault();
        saveBoxes();
        return;
    }

    if (e.key === 'ArrowRight' || e.key === 'n') {
        e.preventDefault();
        navigateTo(PAGE_IDX + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'p') {
        e.preventDefault();
        navigateTo(PAGE_IDX - 1);
    }
});

function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 1500);
}

function saveBoxes() {
    fetch('/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ idx: PAGE_IDX, boxes: boxes, img_w: imgW, img_h: imgH }),
    }).then(r => r.json()).then(data => {
        showToast('Saved ' + boxes.length + ' boxes');
    });
}

function navigateTo(idx) {
    if (idx >= 0 && idx < TOTAL_PAGES) {
        window.location.href = '/annotate?idx=' + idx;
    }
}

function doPrefill() {
    showToast('Running model...');
    fetch('/prefill?idx=' + PAGE_IDX).then(r => r.json()).then(data => {
        if (data.boxes.length === 0) {
            showToast('No detections');
            return;
        }
        boxes = data.boxes;
        selectedIdx = -1;
        render();
        showToast('Prefilled ' + data.boxes.length + ' boxes');
    });
}

function clearBoxes() {
    boxes = [];
    selectedIdx = -1;
    render();
}

document.querySelector('.canvas-wrap').addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    zoom = Math.min(5, Math.max(0.2, zoom * delta));
    applyTransform();
}, { passive: false });
"""


def _build_annotate_html(
    filename, idx, total, n_annotated, initial_boxes, prefill_enabled
):
    """Build the annotation page HTML."""
    classes_dict = {int(k): v for k, v in CLASS_NAMES.items()}
    colors = _generate_colors(len(CLASS_NAMES))
    colors_dict = {i: colors[i] for i in range(len(CLASS_NAMES))}

    boxes_json = json.dumps(initial_boxes)

    label_buttons = ""
    for i, name in enumerate(CLASS_NAMES_LIST):
        color = colors[i]
        active = " active" if i == 0 else ""
        label_buttons += (
            f'<button class="label-btn{active}" '
            f'style="color:{color};" '
            f'onclick="currentLabel={i};updateLabelButtons();'
            f"if(selectedIdx>=0){{boxes[selectedIdx].class_id={i};"
            f'boxes[selectedIdx].label=CLASSES[{i}];render();}}">'
            f'{name}<span class="key">{i + 1}</span></button>'
        )

    prefill_btn = ""
    if prefill_enabled:
        prefill_btn = (
            '<button class="btn btn-prefill" onclick="doPrefill()">'
            'Prefill<span class="shortcut">model</span></button>'
        )

    js = ANNOTATE_JS.replace("INJECT_CLASSES", json.dumps(classes_dict))
    js = js.replace("INJECT_COLORS", json.dumps(colors_dict))
    js = js.replace("INITIAL_BOXES", boxes_json)
    js = js.replace("PAGE_IDX", str(idx))
    js = js.replace("TOTAL_PAGES", str(total))

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Annotate {idx + 1}/{total}</title>
<style>{ANNOTATE_CSS}</style></head><body>
<div class="header">
  <h1>Layout Annotation</h1>
  <div class="header-right">
    <span class="stats" id="box-stats">no boxes</span>
    <span class="stats">annotated={n_annotated}/{total}</span>
    <span class="progress">{idx + 1} / {total}</span>
  </div>
</div>
<div class="main">
  <div class="info"><strong>{filename}</strong></div>
  <div class="canvas-wrap">
    <div class="canvas-container">
      <img id="page-img" src="/image?idx={idx}" />
      <canvas id="overlay"></canvas>
    </div>
    <span id="zoom-info" class="zoom-info">100%</span>
  </div>
</div>
<div class="toolbar">{label_buttons}</div>
<div class="controls">
  <button class="btn btn-nav" onclick="navigateTo({idx - 1})">
    &larr; Prev<span class="shortcut">Left / P</span></button>
  <button class="btn btn-save" onclick="saveBoxes()">
    Save<span class="shortcut">S</span></button>
  {prefill_btn}
  <button class="btn btn-clear" onclick="clearBoxes()">
    Clear</button>
  <button class="btn btn-nav" onclick="navigateTo({idx + 1})">
    Next &rarr;<span class="shortcut">Right / N</span></button>
</div>
<div id="toast" class="toast"></div>
<script>{js}</script>
</body></html>"""


class AnnotationHandler(SimpleHTTPRequestHandler):
    """HTTP handler for layout bounding box annotation."""

    def __init__(self, *args, state=None, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        routes = {
            "/": self._handle_annotate,
            "/annotate": self._handle_annotate,
            "/image": self._handle_image,
            "/prefill": self._handle_prefill,
        }
        handler = routes.get(parsed.path)
        if handler:
            handler(params)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/save":
            self._handle_save()
        else:
            self.send_error(404)

    def _handle_annotate(self, params):
        idx = int(params.get("idx", ["0"])[0])
        idx = max(0, min(idx, len(self.state["images"]) - 1))

        image_path = self.state["images"][idx]
        filename = image_path.name

        existing = next(
            (a for a in self.state["annotations"] if a["filename"] == filename), None
        )
        if existing:
            initial_boxes = existing["boxes"]
        elif self.state["prefill_enabled"]:
            initial_boxes = _prefill_image(image_path, self.state["model"])
        else:
            initial_boxes = []

        html = _build_annotate_html(
            filename,
            idx,
            len(self.state["images"]),
            self.state["n_annotated"],
            initial_boxes,
            self.state["prefill_enabled"],
        )
        self._send_html(html)

    def _handle_image(self, params):
        idx = int(params.get("idx", ["0"])[0])
        if idx < 0 or idx >= len(self.state["images"]):
            self.send_error(404)
            return

        image_path = self.state["images"][idx]
        data = image_path.read_bytes()
        ext = image_path.suffix.lower()
        content_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _handle_prefill(self, params):
        idx = int(params.get("idx", ["0"])[0])
        if idx < 0 or idx >= len(self.state["images"]):
            self._send_json({"boxes": []})
            return

        model = self.state.get("model")
        if model is None:
            self._send_json({"boxes": []})
            return

        image_path = self.state["images"][idx]
        boxes = _prefill_image(image_path, model)
        self._send_json({"boxes": boxes})

    def _handle_save(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        idx = body["idx"]
        boxes = body["boxes"]
        img_w = body["img_w"]
        img_h = body["img_h"]

        image_path = self.state["images"][idx]
        filename = image_path.name

        annotations = self.state["annotations"]
        existing = next((a for a in annotations if a["filename"] == filename), None)
        if existing:
            existing["boxes"] = boxes
            existing["image_width"] = img_w
            existing["image_height"] = img_h
        else:
            annotations.append(
                {
                    "filename": filename,
                    "image_width": img_w,
                    "image_height": img_h,
                    "boxes": boxes,
                    "split": _assign_split(),
                }
            )

        self.state["n_annotated"] = sum(1 for a in annotations if len(a["boxes"]) > 0)
        _save_annotations(annotations)

        self._send_json({"ok": True, "n_boxes": len(boxes)})

    def _send_html(self, html):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def annotate(images_dir, prefill=False, model_path=None, port=8789):
    """Start the annotation server."""
    images = _collect_images(images_dir)
    if not images:
        logger.error(f"No images found in {images_dir}")
        return

    logger.info(f"Found {len(images)} images in {images_dir}")

    annotations = _load_annotations()
    annotated_files = {a["filename"] for a in annotations if len(a["boxes"]) > 0}
    n_annotated = len(annotated_files)
    logger.info(f"Already annotated: {n_annotated}")

    model = None
    if prefill and model_path:
        logger.info(f"Loading prefill model: {model_path}")
        model = _load_model(model_path)

    state = {
        "images": images,
        "annotations": annotations,
        "n_annotated": n_annotated,
        "prefill_enabled": prefill and model is not None,
        "model": model,
    }

    handler = partial(AnnotationHandler, state=state)
    server = HTTPServer(("127.0.0.1", port), handler)

    url = f"http://127.0.0.1:{port}"
    logger.info(f"Annotation server: {url}")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _save_annotations(state["annotations"])
        logger.info(f"Saved {len(state['annotations'])} annotations")


def main():
    """CLI entry point for layout annotation."""
    parser = argparse.ArgumentParser(description="Annotate layout bounding boxes")
    parser.add_argument("images_dir", type=str, help="Directory containing images")
    parser.add_argument("--prefill", action="store_true", help="Enable model prefill")
    parser.add_argument(
        "--model", type=str, default=None, help="Model path for prefill"
    )
    parser.add_argument("--port", type=int, default=8789)
    args = parser.parse_args()

    annotate(
        images_dir=args.images_dir,
        prefill=args.prefill,
        model_path=args.model,
        port=args.port,
    )


if __name__ == "__main__":
    main()
