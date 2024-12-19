from dataclasses import dataclass


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


@dataclass
class LayoutParserTrainingSettings:
    hf_dataset: str = "agomberto/historical-layout"
    data_dir: str = "data"
    model_dir: str = "models"
    model_ft: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    model_ft_name: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    pushed_model_name: str = "test.pt"
    hf_repo: str = "agomberto/historical-layout-ft-test"
    model_ft_dir: str = "yolo_ft"
