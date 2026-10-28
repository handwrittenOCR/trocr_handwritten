import os
import subprocess
import time
import unicodedata
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from trocr_handwritten.htr.settings import HTRSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

ALTO_NS = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}


def _points(attr: str) -> List[tuple]:
    """Parse an ALTO 'x1 y1 x2 y2 ...' attribute into (x, y) tuples."""
    values = [int(float(v)) for v in attr.split()]
    return list(zip(values[0::2], values[1::2]))


def background_color(image: Image.Image) -> tuple:
    """Median RGB colour of the image, used as paper colour for padding and masks."""
    return tuple(int(v) for v in np.median(np.asarray(image).reshape(-1, 3), axis=0))


def read_alto_lines(alto_path: Path) -> List[dict]:
    """
    Read text lines from a Kraken ALTO file, sorted top to bottom.

    Returns:
        List of {id, baseline, polygon, text}; text joins the line's String
        CONTENT values with spaces and is empty for segmentation-only files.
    """
    root = ET.parse(alto_path).getroot()
    lines = []
    for node in root.iterfind(".//a:TextLine", ALTO_NS):
        baseline = _points(node.get("BASELINE", ""))
        shape = node.find("a:Shape/a:Polygon", ALTO_NS)
        polygon = _points(shape.get("POINTS")) if shape is not None else []
        words = [s.get("CONTENT", "") for s in node.iterfind("a:String", ALTO_NS)]
        if baseline and polygon:
            lines.append(
                {
                    "id": node.get("ID"),
                    "baseline": baseline,
                    "polygon": polygon,
                    "text": " ".join(w for w in words if w),
                }
            )
    return sorted(lines, key=lambda line: np.mean([y for _, y in line["baseline"]]))


def batched(items: list, size: int) -> List[list]:
    """Split a list into consecutive chunks of at most `size` items."""
    return [items[i : i + size] for i in range(0, len(items), max(1, size))]


class HTREngine(ABC):
    """
    Base class for line-based HTR engines evaluated at crop level.

    Crops are segmented once with Kraken blla and cached (padded image + ALTO);
    subclasses only recognize the cached lines. `transcribe` assembles the line
    texts top to bottom with newlines and writes one prediction file per crop,
    empty on failure.
    """

    def __init__(self, settings: HTRSettings):
        self.settings = settings
        self.timings: Dict[str, float] = {}

    def run_cli(self, command: List[str], env: Optional[dict] = None) -> str:
        """Run an engine CLI, log stderr tail on failure, and return stdout."""
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
        )
        if result.returncode != 0:
            logger.warning(
                f"{Path(command[0]).name} exited {result.returncode}: "
                f"{result.stderr.strip()[-500:]}"
            )
        return result.stdout

    def cache_paths(self, image_path: Path, input_dir: Path, cache_dir: Path):
        """Return (padded image, ALTO) cache paths mirroring the input subfolders."""
        stem = cache_dir / image_path.relative_to(input_dir).with_suffix("")
        return stem.with_suffix(".png"), stem.with_suffix(".xml")

    def pad_for_segmentation(self, image_path: Path, padded_path: Path) -> None:
        """Pad a wide crop at the bottom so width/height <= seg_max_aspect."""
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        target = max(height, int(width / self.settings.seg_max_aspect))
        canvas = Image.new("RGB", (width, target), background_color(image))
        canvas.paste(image, (0, 0))
        padded_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(padded_path)

    def run_segmenter(self, pairs: List[tuple]) -> None:
        """Segment (padded image, ALTO) pairs in one Kraken blla process."""
        command = [
            self.settings.kraken_bin,
            "--threads",
            str(self.settings.threads),
            "-d",
            self.settings.device,
            "-a",
        ]
        for padded, alto in pairs:
            command += ["-i", str(padded.resolve()), str(alto)]
        self.run_cli(command + ["segment", "-bl"])

    def segment(
        self, images: List[Path], input_dir: Path, cache_dir: Path
    ) -> Dict[Path, Optional[Path]]:
        """
        Segment crops with Kraken blla, reusing cached ALTO files.

        A failed batch is retried crop by crop. Returns image -> ALTO path,
        None when segmentation failed.
        """
        start = time.time()
        pairs = {img: self.cache_paths(img, input_dir, cache_dir) for img in images}
        todo = [img for img, (_, alto) in pairs.items() if not alto.exists()]
        logger.info(f"Segmenting {len(todo)} crops ({len(images) - len(todo)} cached)")
        with tqdm(total=len(todo), desc="Segmenting", unit="crop") as bar:
            for chunk in batched(todo, self.settings.batch_size):
                for img in chunk:
                    self.pad_for_segmentation(img, pairs[img][0])
                self.run_segmenter([pairs[img] for img in chunk])
                missing = [img for img in chunk if not pairs[img][1].exists()]
                if len(chunk) > 1:
                    for img in missing:
                        self.run_segmenter([pairs[img]])
                bar.update(len(chunk))
        self.timings["segment_seconds"] = round(time.time() - start, 1)
        return {
            img: alto if alto.exists() else None for img, (_, alto) in pairs.items()
        }

    @abstractmethod
    def recognize_lines(self, altos: List[Path]) -> Dict[Path, List[str]]:
        """Recognize every line of each segmented crop, in read_alto_lines order."""

    def transcribe(
        self,
        images: List[Path],
        input_dir: Path,
        output_dir: Path,
        cache_dir: Path,
    ) -> List[dict]:
        """
        Segment, recognize and write one NFC prediction per crop (empty on failure).

        Returns per-crop records {image, n_lines, failed}.
        """
        segmented = self.segment(images, input_dir, cache_dir)
        altos = [alto for alto in segmented.values() if alto is not None]
        start = time.time()
        texts: Dict[Path, List[str]] = {}
        with tqdm(total=len(altos), desc="Recognizing", unit="crop") as bar:
            for chunk in batched(altos, self.settings.batch_size):
                texts.update(self.recognize_lines(chunk))
                bar.update(len(chunk))
        self.timings["recognize_seconds"] = round(time.time() - start, 1)

        records = []
        for img, alto in segmented.items():
            lines = texts.get(alto) if alto is not None else None
            text = "\n".join(line.strip() for line in lines or [])
            out = output_dir / img.relative_to(input_dir).with_suffix(
                self.settings.output_extension
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(unicodedata.normalize("NFC", text), encoding="utf-8")
            records.append(
                {
                    "image": str(img),
                    "n_lines": len(lines or []),
                    "failed": lines is None,
                }
            )
        return records
