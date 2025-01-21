from dataclasses import dataclass


@dataclass
class LayoutParserSettings:
    path_folder: str = "data/GUY_politiques__FRANOM22_H2373_IMAGES/raw"
    path_output: str = "data/GUY_politiques__FRANOM22_H2373_IMAGES/processed"
    path_model: str = None
    hf_repo: str = "MarieBgl/historical-layout-ft-test"
    # hf_filename: str = "20241119_v2_yolov10_50_finetuned.pt"
    hf_filename: str = "my_ft_model_20250111.pt"
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
