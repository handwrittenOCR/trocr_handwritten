"""Tests of the ground-truth layout box store."""

import json

from trocr_handwritten.parse.gt_store import (
    boxes_from_metadata,
    gt_path,
    load_gt,
    save_gt,
)


def _metadata(tmp_path):
    meta = {
        "Plein Texte": [
            {
                "cropped_image_name": "000.jpg",
                "coordinates": {"x": 10, "y": 20, "width": 300, "height": 100},
            }
        ],
        "Marge": [
            {
                "cropped_image_name": "001.jpg",
                "coordinates": {"x": 0, "y": 20, "width": 8, "height": 50},
            }
        ],
        "reading_order": [],
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(meta), encoding="utf-8")
    return p


def test_boxes_from_metadata_reads_text_classes_only(tmp_path):
    boxes = boxes_from_metadata(_metadata(tmp_path))
    assert {b["label"] for b in boxes} == {"Plein Texte", "Marge"}
    assert boxes[0]["class_id"] == 4 and boxes[0]["width"] == 300.0


def test_accepted_prefill_vs_drawn(tmp_path):
    yolo = boxes_from_metadata(_metadata(tmp_path))
    gt_dir = tmp_path / "gt"
    p = save_gt(gt_dir, "mar_lamentin", "1848", "pg1", yolo, (1000, 800), yolo)
    assert load_gt(p)["source"] == "yolo_accepted"
    wider = [dict(b, width=b["width"] + 200) for b in yolo]
    save_gt(gt_dir, "mar_lamentin", "1848", "pg1", wider, (1000, 800), yolo)
    rec = load_gt(gt_path(gt_dir, "mar_lamentin", "1848", "pg1"))
    assert rec["source"] == "drawn" and rec["boxes"][0]["width"] == 500


def test_every_version_kept_in_history(tmp_path):
    yolo = boxes_from_metadata(_metadata(tmp_path))
    gt_dir = tmp_path / "gt"
    save_gt(gt_dir, "c", "1840", "pg", yolo, (1000, 800), yolo)
    save_gt(gt_dir, "c", "1840", "pg", [], (1000, 800), yolo)
    hist = list((gt_dir / "_history" / "c" / "1840").glob("*.json"))
    assert len([h for h in hist if h.name.endswith("_previous.json")]) == 1
    assert len(hist) >= 3
