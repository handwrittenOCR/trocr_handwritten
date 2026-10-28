import tempfile
from pathlib import Path
from typing import Dict, List

from trocr_handwritten.htr.engine import HTREngine, read_alto_lines


class KrakenEngine(HTREngine):
    """Kraken recognition (`kraken -f xml ... ocr -m <model>`) on cached blla lines."""

    def recognize_lines(self, altos: List[Path]) -> Dict[Path, List[str]]:
        """Run one Kraken OCR process over the cached ALTO files of a batch."""
        with tempfile.TemporaryDirectory() as tmp:
            outputs = {alto: Path(tmp) / f"{i}.xml" for i, alto in enumerate(altos)}
            command = [
                self.settings.kraken_bin,
                "--threads",
                str(self.settings.threads),
                "-d",
                self.settings.device,
                "-f",
                "xml",
                "-a",
            ]
            for alto, out in outputs.items():
                command += ["-i", str(alto), str(out)]
            self.run_cli(command + ["ocr", "-m", self.settings.model])
            return {
                alto: [line["text"] for line in read_alto_lines(out)]
                for alto, out in outputs.items()
                if out.exists()
            }
