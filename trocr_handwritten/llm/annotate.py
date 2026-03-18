import argparse
import json
import shutil
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from trocr_handwritten.utils.annotation import (
    assign_split,
    collect_images_recursive,
    load_split_annotations,
    save_split_annotations,
    ANNOTATE_BASE_CSS,
)
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

OCR_DIR = Path("data/ocr")


def _get_subfolder(image_path, images_root):
    """
    Extract the subfolder name (class) from an image path relative to images_root.
    e.g. data/processed/images/DOC/Plein Texte/000.jpg → "Plein Texte"

    Args:
        image_path: Full path to the image
        images_root: Root images directory

    Returns:
        str: Subfolder name, or "" if at root
    """
    rel = image_path.relative_to(Path(images_root).resolve())
    parts = rel.parts
    if len(parts) >= 2:
        return parts[-2]
    return ""


def _read_inference(image_path, inference_ext):
    """
    Read the inference file next to an image.

    Args:
        image_path: Path to the image
        inference_ext: Extension of inference files (e.g. ".md")

    Returns:
        str: Inference text, or empty string
    """
    inf_path = image_path.with_suffix(inference_ext)
    if inf_path.exists():
        return inf_path.read_text(encoding="utf-8").strip()
    return ""


OCR_ANNOTATE_CSS = (
    ANNOTATE_BASE_CSS
    + """
.main { flex: 1; display: flex; min-height: 0; overflow: hidden; }
.split-left { flex: 1; display: flex; justify-content: center; align-items: center;
              overflow: auto; padding: 0.5rem; background: #1a1a2e; }
.split-left img { max-width: 100%; max-height: 100%; object-fit: contain; }
.split-right { flex: 1; display: flex; flex-direction: column; padding: 0.5rem;
               background: #0d1117; }
.split-right textarea { flex: 1; background: #161b22; color: #e6edf3; border: 1px solid #30363d;
                        border-radius: 6px; padding: 1rem; font-family: monospace;
                        font-size: 14px; line-height: 1.6; resize: none; outline: none; }
.split-right textarea:focus { border-color: #64ffda; }
.info { font-size: 0.8rem; color: #8892b0; padding: 0.3rem 0.5rem; text-align: center; }
.info strong { color: #ccd6f6; }
"""
)

OCR_ANNOTATE_JS = """
const textarea = document.getElementById('transcription');

function showToast(msg) {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 1500);
}

function saveText() {
    const text = textarea.value;
    fetch('/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ idx: PAGE_IDX, text: text }),
    }).then(r => {
        if (!r.ok) throw new Error(r.status);
        return r.json();
    }).then(data => {
        showToast('Saved');
    }).catch(err => {
        showToast('Error: ' + err.message);
    });
}

function navigateTo(idx) {
    if (idx >= 0 && idx < TOTAL_PAGES) {
        window.location.href = '/annotate?idx=' + idx;
    }
}

document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        saveText();
        return;
    }
    if (e.target.tagName === 'TEXTAREA') return;

    if (e.key === 'ArrowRight' || e.key === 'n') {
        e.preventDefault();
        navigateTo(PAGE_IDX + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'p') {
        e.preventDefault();
        navigateTo(PAGE_IDX - 1);
    }
});

textarea.focus();
"""


def _build_annotate_html(filename, subfolder, idx, total, n_annotated, initial_text):
    """Build the OCR annotation page HTML."""
    text_escaped = (
        initial_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )

    js = OCR_ANNOTATE_JS.replace("PAGE_IDX", str(idx)).replace(
        "TOTAL_PAGES", str(total)
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>OCR Annotate {idx + 1}/{total}</title>
<style>{OCR_ANNOTATE_CSS}</style></head><body>
<div class="header">
  <h1>OCR Annotation</h1>
  <div class="header-right">
    <span class="stats">annotated={n_annotated}/{total}</span>
    <span class="progress">{idx + 1} / {total}</span>
  </div>
</div>
<div class="info"><strong>{filename}</strong> &mdash; {subfolder}</div>
<div class="main">
  <div class="split-left">
    <img src="/image?idx={idx}" />
  </div>
  <div class="split-right">
    <textarea id="transcription">{text_escaped}</textarea>
  </div>
</div>
<div class="controls">
  <button class="btn btn-nav" onclick="navigateTo({idx - 1})">
    &larr; Prev<span class="shortcut">Left / P</span></button>
  <button class="btn btn-save" onclick="saveText()">
    Save<span class="shortcut">Ctrl+S</span></button>
  <button class="btn btn-clear" onclick="document.getElementById('transcription').value=''">
    Clear</button>
  <button class="btn btn-nav" onclick="navigateTo({idx + 1})">
    Next &rarr;<span class="shortcut">Right / N</span></button>
</div>
<div id="toast" class="toast"></div>
<script>{js}</script>
</body></html>"""


class OCRAnnotationHandler(SimpleHTTPRequestHandler):
    """HTTP handler for OCR text annotation."""

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
        subfolder = _get_subfolder(image_path, self.state["images_root"])

        existing = next(
            (
                a
                for a in self.state["annotations"]
                if a["filename"] == filename and a.get("subfolder") == subfolder
            ),
            None,
        )
        if existing:
            initial_text = existing["text"]
        else:
            initial_text = _read_inference(image_path, self.state["inference_ext"])

        html = _build_annotate_html(
            filename,
            subfolder,
            idx,
            len(self.state["images"]),
            self.state["n_annotated"],
            initial_text,
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

    def _handle_save(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        idx = body["idx"]
        text = body["text"].strip()

        image_path = self.state["images"][idx]
        filename = image_path.name
        subfolder = _get_subfolder(image_path, self.state["images_root"])

        annotations = self.state["annotations"]
        existing = next(
            (
                a
                for a in annotations
                if a["filename"] == filename and a.get("subfolder") == subfolder
            ),
            None,
        )

        if existing:
            existing["text"] = text
        else:
            split = assign_split()
            annotations.append(
                {
                    "filename": filename,
                    "subfolder": subfolder,
                    "text": text,
                    "split": split,
                }
            )

        split = next(
            a["split"]
            for a in annotations
            if a["filename"] == filename and a.get("subfolder") == subfolder
        )
        dst_images = OCR_DIR / split / "images" / subfolder
        dst_labels = OCR_DIR / split / "labels" / subfolder
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_path, dst_images / filename)
        label_path = dst_labels / (image_path.stem + ".txt")
        label_path.write_text(text, encoding="utf-8")

        self.state["n_annotated"] = sum(1 for a in annotations if a.get("text"))
        save_split_annotations(annotations, OCR_DIR)

        self._send_json({"ok": True})

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


def annotate(images_dir, inference_ext=".md", import_dir=None, port=8790):
    """
    Start the OCR annotation server.

    Args:
        images_dir: Directory with processed images (document/class/image.jpg)
        inference_ext: Extension of inference files to pre-fill
        import_dir: Optional legacy data directory to import first
        port: Server port
    """
    if import_dir:
        from trocr_handwritten.utils.annotation import import_legacy_annotations

        import_legacy_annotations(import_dir, str(OCR_DIR), logger)

    images_root = Path(images_dir).resolve()
    images = collect_images_recursive(images_dir)
    if not images:
        logger.error(f"No images found in {images_dir}")
        return

    logger.info(f"Found {len(images)} images in {images_dir}")

    annotations = load_split_annotations(OCR_DIR)
    n_annotated = sum(1 for a in annotations if a.get("text"))
    logger.info(f"Already annotated: {n_annotated}")

    state = {
        "images": images,
        "images_root": images_root,
        "annotations": annotations,
        "n_annotated": n_annotated,
        "inference_ext": inference_ext,
    }

    handler = partial(OCRAnnotationHandler, state=state)
    server = HTTPServer(("127.0.0.1", port), handler)

    url = f"http://127.0.0.1:{port}"
    logger.info(f"OCR annotation server: {url}")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        save_split_annotations(state["annotations"], OCR_DIR)
        logger.info(f"Saved {len(state['annotations'])} annotations")


def main():
    """CLI entry point for OCR annotation."""
    parser = argparse.ArgumentParser(description="Annotate OCR transcriptions")
    parser.add_argument("images_dir", type=str, help="Directory with processed images")
    parser.add_argument("--inference-ext", type=str, default=".md")
    parser.add_argument(
        "--import-dir",
        type=str,
        default=None,
        help="Import legacy annotated data first",
    )
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()

    annotate(
        images_dir=args.images_dir,
        inference_ext=args.inference_ext,
        import_dir=args.import_dir,
        port=args.port,
    )


if __name__ == "__main__":
    main()
