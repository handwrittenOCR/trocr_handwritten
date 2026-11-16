from dataclasses import dataclass, field

CLASS_NAMES = {
    "0": "Title",
    "1": "Bruit",
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
    path_output: str = "data/processed/images/"
    path_model: str = None
    hf_repo: str = "agomberto/historical-layout-rfdetr-m"
    hf_filename: str = "20261113_rfdetr_medium_layout.pt"
    device: str = "cpu"
    conf: float = 0.2
    iou: float = 0.5
    batch_size: int = 32
    create_annotation_json: bool = True
    class_names: dict = field(default_factory=lambda: dict(CLASS_NAMES))


@dataclass
class TrainingSettings:
    path_data: str = "data/layout"
    images_dir: str = "data/raw/images"
    model_base: str = "yolo11n.pt"
    epochs: int = 100
    imgsz: int = 1024
    batch: int = 8
    device: str = "auto"
    workers: int = 4
    patience: int = 20
    freeze: int = 0
    dropout: float = 0.0
    weight_decay: float = 0.0005
    run_name: str = "layout"
    class_names: dict = field(default_factory=lambda: dict(CLASS_NAMES))


@dataclass
class FasterRCNNSettings(TrainingSettings):
    model_base: str = "fasterrcnn_resnet50_fpn"
    pretrained: bool = True
    trainable_backbone_layers: int = 3
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    warmup_iters: int = 1000
    hflip: float = 0.5
    max_det: int = 300
    seed: int = 0


@dataclass
class RFDETRSettings(TrainingSettings):
    model_base: str = "rfdetr_medium"
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    seed: int = 0


@dataclass
class EvaluationSettings:
    path_data: str = "data/layout"
    path_model: str = None
    hf_repo: str = "agomberto/historical-layout-rfdetr-m"
    hf_filename: str = "20261113_rfdetr_medium_layout.pt"
    split: str = "test"
    conf: float = 0.2
    ap_conf: float = 0.001
    iou: float = 0.5
    imgsz: int = 1024
    device: str = "cpu"
    out_json: str = None
    iou_thresholds_ap: tuple = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    iou_thresholds_prf: tuple = (0.5, 0.7)
    confusion_iou: float = 0.5
    ap_recall_points: int = 101
    conf_sweep: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    class_names: dict = field(default_factory=lambda: dict(CLASS_NAMES))


@dataclass
class LayoutDatasetSettings:
    repo_id: str = "agomberto/historical-layout-colonial"
    out_dir: str = "data/layout_colonial"
    splits: tuple = ("train", "dev", "test")
    class_names: dict = field(default_factory=lambda: dict(CLASS_NAMES))


@dataclass
class BenchmarkSettings:
    path_data: str = "data/layout_merged"
    images_dir: str = "data/layout_merged/images_src"
    device: str = "cuda"
    splits: tuple = ("dev", "test")
    doclayout_repo: str = "juliozhao/DocLayout-YOLO-DocSynth300K-pretrain"
    doclayout_filename: str = "doclayout_yolo_docsynth300k_imgsz1600.pt"
    weights_dir: str = "models"
    runs: dict = field(
        default_factory=lambda: {
            "bench_yolo11m_v2": "yolo11m.pt",
            "bench_yolo26m": "yolo26m.pt",
            "bench_doclayout": "doclayout",
            "bench_rfdetr_m": "rfdetr_medium",
            "bench_rtdetr_l": "rtdetr-l.pt",
            "bench_frcnn": "fasterrcnn_resnet50_fpn",
        }
    )
    extra_args: dict = field(
        default_factory=lambda: {"bench_frcnn": ["--patience", "10"]}
    )
