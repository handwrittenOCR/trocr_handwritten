import os
from typing import Literal, Optional

from pydantic import BaseModel, Field


class HTRSettings(BaseModel):
    """Configuration for line-based HTR engines (Kraken, PyLaia) run zero-shot."""

    engine: Literal["kraken", "pylaia"] = Field(
        default="kraken", description="Recognition engine."
    )
    model: str = Field(
        description="Kraken: path to the .mlmodel. PyLaia: directory holding "
        "model, weights.ckpt and syms.txt."
    )
    device: str = Field(default="cpu", description="cpu, cuda or cuda:N.")
    batch_size: int = Field(
        default=8, description="Crops per engine subprocess (one model load each)."
    )
    threads: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="OpenMP/BLAS threads for Kraken (all cores by default).",
    )
    kraken_bin: str = Field(
        default=".venv-kraken/bin/kraken", description="Kraken CLI executable."
    )
    pylaia_decode_bin: str = Field(
        default=".venv-pylaia/bin/pylaia-htr-decode-ctc",
        description="PyLaia decoding CLI executable.",
    )
    pylaia_lib_dir: str = Field(
        default=".venv-pylaia/lib/gomp",
        description="Prepended to LD_LIBRARY_PATH for PyLaia: holds a libgomp.so.1 "
        "link to torch's bundled copy (aarch64 wheels miss the system one).",
    )
    seg_cache_dir: Optional[str] = Field(
        default=None,
        description="Segmentation cache shared by both engines. "
        "Defaults to <input_dir>/../segmentation.",
    )
    seg_max_aspect: float = Field(
        default=2.0,
        description="Crops wider than this width/height ratio are padded at the "
        "bottom before blla, which rescales to 1800 px height and runs out of "
        "memory on wide crops. Bottom padding keeps line coordinates unchanged.",
    )
    line_height: int = Field(
        default=128, description="PyLaia fixed input line height in pixels."
    )
    output_extension: str = Field(default=".md", description="Prediction extension.")
