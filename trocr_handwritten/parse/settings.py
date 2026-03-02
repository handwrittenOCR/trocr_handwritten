from dataclasses import dataclass


@dataclass
class LayoutParserSettings:
    path_folder: str = "data/raw/images"
    # in the case that preprocessing works, modify to "data/raw/images/preprocess"
    path_output: str = "data/processed/images"
    path_model: str = "models/yolo_layout_model/20250111_yolov10_bagnards_EC.pt"
    hf_repo: str = None
    hf_filename: str = None
    device: str = "cpu"
    conf: float = 0.2
    iou: float = 0.5
    create_annotation_json: bool = True

    def __post_init__(self):
        self.class_names = {
            "0": "Title",
            "1": "En-tête",
            "2": "Marge",
            "3": "Nom",
            "4": "Plein Texte",
            "5": "Signature",
            "6": "Table",
            "7": "Section",
        }
