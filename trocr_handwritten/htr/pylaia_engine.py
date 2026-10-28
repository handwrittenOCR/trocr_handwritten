import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

from trocr_handwritten.htr.engine import (
    HTREngine,
    background_color,
    read_alto_lines,
)


class PyLaiaEngine(HTREngine):
    """
    PyLaia recognition (`pylaia-htr-decode-ctc`, greedy CTC, no language model)
    on the cached Kraken lines, resized to a fixed height.
    """

    def extract_line(
        self, image: Image.Image, polygon: List[tuple]
    ) -> Optional[Image.Image]:
        """
        Crop a line polygon, fill outside it with paper colour, resize to
        line_height. Returns None for a degenerate polygon.
        """
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        filled = Image.new("RGB", image.size, background_color(image))
        filled.paste(image, (0, 0), mask)
        box = mask.getbbox()
        if box is None:
            return None
        line = filled.crop(box).convert("L")
        width, height = line.size
        target = self.settings.line_height
        return line.resize((max(1, round(width * target / height)), target))

    def recognize_lines(self, altos: List[Path]) -> Dict[Path, List[str]]:
        """Extract every line of the batch and decode them in one PyLaia process."""
        model_dir = Path(self.settings.model)
        with tempfile.TemporaryDirectory() as tmp:
            ids, saved = {}, []
            for i, alto in enumerate(altos):
                image = Image.open(alto.with_suffix(".png")).convert("RGB")
                ids[alto] = []
                for j, line in enumerate(read_alto_lines(alto)):
                    line_id = f"{i}_{j}"
                    ids[alto].append(line_id)
                    crop = self.extract_line(image, line["polygon"])
                    if crop is not None:
                        crop.save(Path(tmp) / f"{line_id}.png")
                        saved.append(line_id)
            if not saved:
                return {alto: [""] * len(v) for alto, v in ids.items()}
            img_list = Path(tmp) / "img_list.txt"
            img_list.write_text("\n".join(saved) + "\n")
            gpus = "0" if self.settings.device == "cpu" else "1"
            stdout = self.run_cli(
                [
                    self.settings.pylaia_decode_bin,
                    str(model_dir / "syms.txt"),
                    str(img_list),
                    "--img_dirs",
                    f"[{tmp}]",
                    "--common.train_path",
                    str(model_dir),
                    "--common.model_filename",
                    "model",
                    "--common.checkpoint",
                    "weights.ckpt",
                    "--common.experiment_dirname",
                    ".",
                    "--decode.join_string",
                    "",
                    "--decode.convert_spaces",
                    "true",
                    "--data.batch_size",
                    str(self.settings.batch_size),
                    "--trainer.gpus",
                    gpus,
                ],
                env={
                    "LD_LIBRARY_PATH": os.pathsep.join(
                        [
                            str(Path(self.settings.pylaia_lib_dir).resolve()),
                            os.environ.get("LD_LIBRARY_PATH", ""),
                        ]
                    )
                },
            )
        decoded = dict(
            (line.split(" ", 1) + [""])[:2] for line in stdout.splitlines() if line
        )
        if not decoded:
            return {}
        return {alto: [decoded.get(i, "") for i in v] for alto, v in ids.items()}
