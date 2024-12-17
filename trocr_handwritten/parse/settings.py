from dataclasses import dataclass
import logging


@dataclass
class LayoutParserSettings:
    path_folder: str = "data/raw/images"
    path_output: str = "data/processed/images/"
    path_model: str = None
    hf_repo: str = "agomberto/historical-layout-ft"
    hf_filename: str = "20241119_v2_yolov10_50_finetuned.pt"
    device: str = "cpu"
    conf: float = 0.2
    iou: float = 0.5
    create_annotation_json: bool = True
    logger: logging.Logger = None

    def __post_init__(self):
        self.class_names = {
            "0": "Title",
            "1": "En-tête",
            "2": "Marge",
            "3": "Nom",
            "4": "Plein Texte",
            "5": "Signature",
            "6": "Table",
        }
