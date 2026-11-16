import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from trocr_handwritten.parse.settings import LayoutDatasetSettings
from trocr_handwritten.parse.utils import _get_hf_token
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


def convert_layout_snapshot(snapshot_dir, out_dir, class_names, splits):
    """
    Turn a downloaded layout dataset (imagefolder: layout/<split>/images/ +
    metadata.jsonl) into the layout that parse.train reads:
    <out_dir>/<split>/annotations.json and <out_dir>/images_src/ (relative symlinks).

    Args:
        snapshot_dir: Local copy of the Hugging Face dataset repository
        out_dir: Output directory, used as --path-data for parse.train
        class_names: {"0": "Title", ...} class map of the settings
        splits: Split folder names to convert

    Returns:
        dict: {split: number of pages}
    """
    snapshot_dir, out_dir = Path(snapshot_dir), Path(out_dir)
    class_ids = {name: int(i) for i, name in class_names.items()}
    images_src = out_dir / "images_src"
    images_src.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in splits:
        split_dir = snapshot_dir / "layout" / split
        entries = []
        for line in (
            (split_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            row = json.loads(line)
            src = split_dir / row["file_name"]
            name = Path(row["file_name"]).name
            link = images_src / name
            if not link.is_symlink():
                link.symlink_to(os.path.relpath(src, images_src))
            entries.append(
                {
                    "filename": name,
                    "image_width": row["width"],
                    "image_height": row["height"],
                    "register": row["register"],
                    "source": row["source"],
                    "boxes": [
                        {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "class_id": class_ids[c],
                            "label": c,
                        }
                        for (x, y, w, h), c in zip(
                            row["objects"]["bbox"], row["objects"]["category"]
                        )
                    ],
                }
            )
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        (out_dir / split / "annotations.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        counts[split] = len(entries)
    return counts


def main():
    """
    CLI: download the layout benchmark dataset from Hugging Face and prepare it for
    parse.train and parse.metrics (--path-data <out_dir> --images-dir <out_dir>/images_src).
    """
    defaults = LayoutDatasetSettings()
    parser = argparse.ArgumentParser(
        description="Download and prepare the layout benchmark dataset"
    )
    parser.add_argument("--repo-id", default=defaults.repo_id)
    parser.add_argument("--out-dir", default=defaults.out_dir)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    snapshot = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=out_dir / "hf",
        allow_patterns=["layout/**", "README.md"],
        token=_get_hf_token(),
    )
    counts = convert_layout_snapshot(
        snapshot, out_dir, defaults.class_names, defaults.splits
    )
    logger.info(f"Prepared {out_dir}: {counts}")
    logger.info(
        f"Train: python -m trocr_handwritten.parse.train --path-data {out_dir} --images-dir {out_dir}/images_src"
    )


if __name__ == "__main__":
    main()
