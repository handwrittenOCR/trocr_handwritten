from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def region_crops(manifest, images_dir):
    """
    Yield the grayscale crop of every region of a manifest, page by page.

    Args:
        manifest: DataFrame with columns page, region_id, x, y, w, h.
        images_dir: Folder of the page images.

    Yields:
        tuple: (region row, grayscale crop, page width, page height).
    """
    for page, g in manifest.groupby("page"):
        im = cv2.imread(str(Path(images_dir) / page), cv2.IMREAD_GRAYSCALE)
        H, W = im.shape
        for r in g.itertuples():
            x0, y0 = max(0, int(r.x)), max(0, int(r.y))
            yield r, im[y0 : int(r.y + r.h), x0 : int(r.x + r.w)], W, H


def measures(gray):
    """
    Ink, connected component and stroke measures of a grayscale crop.

    Args:
        gray: Grayscale crop (uint8 array).

    Returns:
        dict: One value per measure.
    """
    s = 512 / max(gray.shape)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bg = float(np.median(gray))
    ink = (gray < bg - 40).astype(np.uint8)
    _, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    st = st[1:]
    st = st[st[:, cv2.CC_STAT_AREA] >= 4]
    if len(st):
        hs, ws, ar = (
            st[:, cv2.CC_STAT_HEIGHT],
            st[:, cv2.CC_STAT_WIDTH],
            st[:, cv2.CC_STAT_AREA],
        )
    else:
        hs = ws = ar = np.zeros(1)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)[ink > 0]
    rows = ink.mean(axis=1)
    area = gray.size
    return {
        "ink": ink.mean(),
        "otsu": otsu.mean(),
        "bg": bg,
        "contrast": bg - np.percentile(gray, 2),
        "std": gray.std(),
        "cc_n": len(st) / area * 1e4,
        "cc_h_mean": hs.mean(),
        "cc_h_cv": hs.std() / (hs.mean() + 1e-6),
        "cc_w_mean": ws.mean(),
        "cc_w_cv": ws.std() / (ws.mean() + 1e-6),
        "cc_big": (ar > 0.01 * area).mean(),
        "cc_area_med": np.median(ar),
        "stroke_mean": dist.mean() if dist.size else 0,
        "stroke_cv": dist.std() / (dist.mean() + 1e-6) if dist.size else 0,
        "rows_cv": rows.std() / (rows.mean() + 1e-6),
        "rows_ink": (rows > 0.02).mean(),
    }


def measure_table(manifest, images_dir, cache):
    """
    One row of measures and box geometry per region, cached as parquet.

    Args:
        manifest: DataFrame of regions (see region_crops).
        images_dir: Folder of the page images.
        cache: Parquet path, read if it exists.

    Returns:
        DataFrame: region_id and one column per measure.
    """
    if Path(cache).exists():
        return pd.read_parquet(cache)
    out = []
    for r, crop, W, H in tqdm(
        region_crops(manifest, images_dir), total=len(manifest), desc="mesures"
    ):
        f = measures(crop) if crop.size else {}
        f.update(
            region_id=r.region_id,
            w=r.w / W,
            h=r.h / H,
            aspect=np.log((r.w + 1) / (r.h + 1)),
            cx=(r.x + r.w / 2) / W,
            cy=(r.y + r.h / 2) / H,
        )
        out.append(f)
    df = pd.DataFrame(out)
    df.to_parquet(cache)
    return df


def square(gray, size):
    """
    Pad a grayscale crop to a square with its median colour and resize it.

    Args:
        gray: Grayscale crop (uint8 array), possibly empty.
        size: Output side in pixels.

    Returns:
        PIL.Image: RGB square image.
    """
    if not gray.size:
        gray = np.full((8, 8), 255, np.uint8)
    side = max(gray.shape)
    sq = np.full((side, side), int(np.median(gray)), np.uint8)
    y0, x0 = (side - gray.shape[0]) // 2, (side - gray.shape[1]) // 2
    sq[y0 : y0 + gray.shape[0], x0 : x0 + gray.shape[1]] = gray
    return Image.fromarray(
        cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    ).convert("RGB")


class Embedder:
    """
    Frozen image encoder from Hugging Face (DINOv2, ResNet, SigLIP or TrOCR encoder).

    Crops are padded to a square and resized before the model processor. The vector
    is the image features when the model has them, the mean of the last hidden
    states for TrOCR, the pooled output otherwise.
    """

    def __init__(self, name, image_size=448, batch_size=32):
        """
        Store the settings; the model is loaded at the first forward pass, so cached embeddings need no download.

        Args:
            name: Hugging Face model id.
            image_size: Side of the square crop given to the processor.
            batch_size: Crops per forward pass.
        """
        self.name, self.image_size, self.batch_size = name, image_size, batch_size
        self.model = self.proc = None

    def _load(self):
        """Load the model and its processor."""
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            VisionEncoderDecoderModel,
        )

        model = (
            VisionEncoderDecoderModel.from_pretrained(self.name).encoder
            if "trocr" in self.name
            else AutoModel.from_pretrained(self.name)
        )
        self.model = model.eval()
        self.proc = AutoImageProcessor.from_pretrained(self.name)

    @property
    def short(self):
        """Short model name used in column and file names (dinov2, resnet, trocr...)."""
        return self.name.split("/")[-1].split("-")[0]

    def _forward(self, images):
        """Embed a list of PIL images."""
        import torch

        if self.model is None:
            self._load()
        with torch.no_grad():
            px = self.proc(images=images, return_tensors="pt")["pixel_values"]
            if hasattr(self.model, "get_image_features"):
                v = self.model.get_image_features(pixel_values=px)
                v = getattr(v, "pooler_output", v)
            else:
                o = self.model(pixel_values=px)
                v = (
                    o.last_hidden_state.mean(1)
                    if "trocr" in self.name
                    else o.pooler_output
                )
        return v.flatten(1).numpy()

    def embed(self, crops, total=None):
        """
        Embed grayscale crops batch by batch.

        Args:
            crops: Iterable of grayscale crops.
            total: Number of crops, for the progress bar.

        Returns:
            np.ndarray: One row per crop.
        """
        vecs, buf = [], []
        for crop in tqdm(crops, total=total, desc=self.name):
            buf.append(square(crop, self.image_size))
            if len(buf) == self.batch_size:
                vecs.append(self._forward(buf))
                buf = []
        if buf:
            vecs.append(self._forward(buf))
        return np.concatenate(vecs)


def embedding_table(manifest, images_dir, embedder, cache):
    """
    One embedding per region, cached as parquet.

    Args:
        manifest: DataFrame of regions (see region_crops).
        images_dir: Folder of the page images.
        embedder: Embedder instance.
        cache: Parquet path, read if it exists.

    Returns:
        DataFrame: region_id and columns e0..eN.
    """
    if Path(cache).exists():
        return pd.read_parquet(cache)
    ids = []

    def crops():
        """Crops in manifest order, region ids recorded on the way."""
        for r, crop, _, _ in region_crops(manifest, images_dir):
            ids.append(r.region_id)
            yield crop

    v = embedder.embed(crops(), total=len(manifest))
    df = pd.DataFrame(v, columns=[f"e{i}" for i in range(v.shape[1])]).assign(
        region_id=ids
    )
    df.to_parquet(cache)
    return df
