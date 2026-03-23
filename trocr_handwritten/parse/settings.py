from dataclasses import dataclass, field

CLASS_NAMES = {
    "0": "Title",
    "1": "En-tête",
    "2": "Marge",
    "3": "Nom",
    "4": "Plein Texte",
    "5": "Signature",
    "6": "Table",
    "7": "Section",
}

CLASS_NAMES_LIST = [CLASS_NAMES[str(i)] for i in range(len(CLASS_NAMES))]


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
    class_names: dict = field(default_factory=lambda: dict(CLASS_NAMES))


@dataclass
class TrainingSettings:
    path_data: str = "data/layout"
    model_base: str = "yolo11n.pt"
    epochs: int = 50
    imgsz: int = 1024
    batch: int = 8
    device: str = "cpu"
    patience: int = 20
    freeze: int = 0
    dropout: float = 0.0
    weight_decay: float = 0.0005
    run_name: str = "layout"
    class_names: dict = field(default_factory=lambda: dict(CLASS_NAMES))


@dataclass
class EvaluationSettings:
    path_data: str = "data/layout"
    path_model: str = None
    hf_repo: str = "agomberto/historical-layout-ft"
    hf_filename: str = "20241119_v2_yolov10_50_finetuned.pt"
    split: str = "test"
    conf: float = 0.2
    iou: float = 0.5
    imgsz: int = 1024
    device: str = "cpu"
    class_names: dict = field(default_factory=lambda: dict(CLASS_NAMES))
