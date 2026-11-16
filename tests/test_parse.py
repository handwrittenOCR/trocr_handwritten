import pytest
from unittest.mock import Mock, patch

from trocr_handwritten.parse.settings import (
    LayoutParserSettings,
    TrainingSettings,
    EvaluationSettings,
    CLASS_NAMES,
    CLASS_NAMES_LIST,
)
from trocr_handwritten.parse.utils import calculate_iou, YOLOModel


class TestLayoutParserSettings:
    """Tests for LayoutParserSettings."""

    def test_default_values(self):
        """Test default settings values."""
        settings = LayoutParserSettings()
        assert settings.path_folder == "data/raw/images"
        assert settings.path_output == "data/processed/images/"
        assert settings.device == "cpu"
        assert settings.conf == 0.2
        assert settings.iou == 0.5

    def test_class_names_initialized(self):
        """Test that class_names are initialized from CLASS_NAMES."""
        settings = LayoutParserSettings()
        assert hasattr(settings, "class_names")
        assert "0" in settings.class_names
        assert settings.class_names["0"] == "Title"
        assert settings.class_names["4"] == "Plein Texte"

    def test_custom_values(self):
        """Test custom settings values."""
        settings = LayoutParserSettings(
            path_folder="custom/path",
            device="cuda",
            conf=0.5,
        )
        assert settings.path_folder == "custom/path"
        assert settings.device == "cuda"
        assert settings.conf == 0.5


class TestTrainingSettings:
    """Tests for TrainingSettings."""

    def test_default_values(self):
        """Test default training settings."""
        settings = TrainingSettings()
        assert settings.model_base == "yolo11n.pt"
        assert settings.epochs == 100
        assert settings.imgsz == 1024
        assert settings.batch == 8
        assert settings.class_names == CLASS_NAMES

    def test_custom_values(self):
        """Test custom training settings."""
        settings = TrainingSettings(epochs=150, batch=16, model_base="yolo11s.pt")
        assert settings.epochs == 150
        assert settings.batch == 16
        assert settings.model_base == "yolo11s.pt"


class TestEvaluationSettings:
    """Tests for EvaluationSettings."""

    def test_default_values(self):
        """Test default evaluation settings."""
        settings = EvaluationSettings()
        assert settings.split == "test"
        assert settings.conf == 0.2
        assert settings.iou == 0.5
        assert settings.class_names == CLASS_NAMES


class TestClassNames:
    """Tests for CLASS_NAMES constants."""

    def test_class_names_has_8_classes(self):
        """Test that CLASS_NAMES has 8 classes."""
        assert len(CLASS_NAMES) == 8

    def test_class_names_list_ordered(self):
        """Test that CLASS_NAMES_LIST is correctly ordered."""
        assert CLASS_NAMES_LIST[0] == "Title"
        assert CLASS_NAMES_LIST[7] == "Section"
        assert len(CLASS_NAMES_LIST) == 8


class TestCalculateIou:
    """Tests for the calculate_iou function."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes is 1.0."""
        box = [0, 0, 100, 100]
        iou = calculate_iou(box, box)
        assert iou == 1.0

    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0.0."""
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 50, 50]
        iou = calculate_iou(box1, box2)
        assert iou == 0.0

    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 100, 100]
        iou = calculate_iou(box1, box2)
        assert 0 < iou < 1

    def test_contained_box(self):
        """Test IoU when one box is contained in another."""
        box1 = [0, 0, 100, 100]
        box2 = [25, 25, 50, 50]
        iou = calculate_iou(box1, box2)
        expected = (50 * 50) / (100 * 100 + 50 * 50 - 50 * 50)
        assert abs(iou - expected) < 0.01

    def test_touching_boxes(self):
        """Test IoU of boxes that just touch (edge to edge)."""
        box1 = [0, 0, 50, 50]
        box2 = [50, 0, 50, 50]
        iou = calculate_iou(box1, box2)
        assert iou == 0.0


class TestYOLOModel:
    """Tests for the YOLOModel class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = LayoutParserSettings(
            hf_repo="test/repo",
            hf_filename="model.pt",
        )
        return settings

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @patch("trocr_handwritten.parse.utils._get_hf_token", return_value=None)
    @patch("trocr_handwritten.parse.utils._detect_model_backend")
    @patch("trocr_handwritten.parse.utils.hf_hub_download")
    @patch("trocr_handwritten.parse.utils._load_model")
    def test_load_from_hf(
        self,
        mock_load,
        mock_download,
        mock_detect,
        mock_token,
        mock_settings,
        mock_logger,
    ):
        """Test loading model from HuggingFace Hub."""
        mock_download.return_value = "/path/to/model.pt"
        mock_detect.return_value = "doclayout"
        mock_load.return_value = Mock()

        YOLOModel(mock_settings, mock_logger)

        mock_download.assert_called_once_with(
            repo_id="test/repo",
            filename="model.pt",
            token=None,
        )
        mock_load.assert_called_once_with("/path/to/model.pt")

    @patch("trocr_handwritten.parse.utils._detect_model_backend")
    @patch("trocr_handwritten.parse.utils._load_model")
    def test_load_from_local(self, mock_load, mock_detect, mock_logger):
        """Test loading model from local path."""
        settings = LayoutParserSettings(
            path_model="/local/model.pt",
            hf_repo=None,
        )
        mock_detect.return_value = "ultralytics"
        mock_load.return_value = Mock()

        YOLOModel(settings, mock_logger)

        mock_load.assert_called_once_with("/local/model.pt")

    def test_raises_without_model_source(self, mock_logger):
        """Test that error is raised when no model source provided."""
        settings = LayoutParserSettings(
            path_model=None,
            hf_repo=None,
            hf_filename=None,
        )

        with pytest.raises(ValueError, match="Either path_model or hf_repo"):
            YOLOModel(settings, mock_logger)

    @patch("trocr_handwritten.parse.utils._get_hf_token", return_value=None)
    @patch("trocr_handwritten.parse.utils._detect_model_backend")
    @patch("trocr_handwritten.parse.utils.hf_hub_download")
    @patch("trocr_handwritten.parse.utils._load_model")
    def test_predict_calls_model(
        self,
        mock_load,
        mock_download,
        mock_detect,
        mock_token,
        mock_settings,
        mock_logger,
    ):
        """Test that predict calls the underlying model."""
        mock_download.return_value = "/path/to/model.pt"
        mock_detect.return_value = "ultralytics"
        mock_model_instance = Mock(spec=["predict"])
        mock_load.return_value = mock_model_instance

        model = YOLOModel(mock_settings, mock_logger)
        model.predict("/test/folder")

        mock_model_instance.predict.assert_called_once_with(
            "/test/folder",
            imgsz=1024,
            conf=mock_settings.conf,
            iou=mock_settings.iou,
            device=mock_settings.device,
        )


class TestRFDETR:
    """Tests for the RF-DETR settings, dataset layout, backend detection and detector."""

    @pytest.fixture
    def yolo_data(self, tmp_path):
        """YOLO train/dev/test splits with one labelled page each, plus an unlabelled dev page."""
        for split in ["train", "dev", "test"]:
            (tmp_path / split / "images").mkdir(parents=True)
            (tmp_path / split / "labels").mkdir(parents=True)
            (tmp_path / split / "images" / f"{split}_p1.jpg").write_bytes(b"jpg")
            (tmp_path / split / "labels" / f"{split}_p1.txt").write_text(
                "3 0.5 0.5 0.2 0.1"
            )
        (tmp_path / "dev" / "images" / "dev_p2.jpg").write_bytes(b"jpg")
        return tmp_path

    def test_settings_defaults(self):
        """RF-DETR keeps the common budget and an imgsz divisible by patch_size * num_windows."""
        from trocr_handwritten.parse.settings import RFDETRSettings

        settings = RFDETRSettings()
        assert settings.model_base == "rfdetr_medium"
        assert (settings.epochs, settings.patience, settings.batch) == (100, 20, 8)
        assert settings.imgsz % (16 * 2) == 0

    def test_build_dataset(self, yolo_data):
        """dev becomes valid, files are relative symlinks to the source splits, data.yaml lists the classes."""
        import os
        import yaml

        from trocr_handwritten.parse.rfdetr import build_rfdetr_dataset

        out = build_rfdetr_dataset(
            yolo_data, yolo_data / "models" / "run" / "dataset", CLASS_NAMES_LIST
        )
        link = out / "valid" / "images" / "dev_p1.jpg"
        assert link.is_symlink() and not os.path.isabs(os.readlink(link))
        assert link.resolve() == (yolo_data / "dev" / "images" / "dev_p1.jpg").resolve()
        assert (out / "valid" / "labels" / "dev_p1.txt").read_text().startswith("3 ")
        assert (out / "valid" / "images" / "dev_p2.jpg").is_symlink()
        assert not (out / "valid" / "labels" / "dev_p2.txt").exists()
        assert not (out / "dev").exists()
        data = yaml.safe_load((out / "data.yaml").read_text())
        assert data["names"] == CLASS_NAMES_LIST and data["val"] == "valid/images"
        build_rfdetr_dataset(yolo_data, out, CLASS_NAMES_LIST)

    def test_detect_backend(self, tmp_path):
        """A checkpoint written by RF-DETR training is routed to RFDETRDetector."""
        import torch

        from trocr_handwritten.parse.rfdetr import RFDETRDetector
        from trocr_handwritten.parse.utils import _detect_model_backend, _load_model

        path = tmp_path / "best.pt"
        torch.save({"model": {}, "args": {}, "model_name": "RFDETRMedium"}, path)
        assert _detect_model_backend(path) == "rfdetr"
        assert isinstance(_load_model(path), RFDETRDetector)

    def test_detector_outputs_yolo_ids(self):
        """detect() returns numpy arrays with 0-based class ids and handles empty pages."""
        import sys
        import types

        import numpy as np

        from trocr_handwritten.parse.rfdetr import RFDETRDetector

        det = Mock()
        det.__len__ = Mock(return_value=2)
        det.xyxy = np.array([[0, 0, 10, 10], [5, 5, 20, 20]], dtype=np.float32)
        det.class_id = np.array([0, 7])
        det.confidence = np.array([0.9, 0.01])
        empty = Mock()
        empty.__len__ = Mock(return_value=0)
        model = Mock()
        model.predict.side_effect = [det, empty]
        fake = types.ModuleType("rfdetr")
        fake.RFDETR = Mock()
        fake.RFDETR.from_checkpoint.return_value = model

        detector = RFDETRDetector("best.pt")
        with patch.dict(sys.modules, {"rfdetr": fake}):
            boxes, classes, confs = detector.detect("p.jpg", 0.001, 0.5, "cpu")
            assert classes.tolist() == [0, 7] and boxes.shape == (2, 4)
            boxes, classes, confs = detector.detect("p.jpg", 0.001, 0.5, "cpu")
            assert boxes.shape == (0, 4) and len(classes) == 0
        fake.RFDETR.from_checkpoint.assert_called_once_with("best.pt", device="cpu")
        _, kwargs = model.predict.call_args
        assert kwargs["threshold"] == 0.001


class TestDetectPipeline:
    """Tests for the inference pipeline with detectors exposing detect()."""

    def test_detection_result_matches_ultralytics_layout(self):
        """DetectionResult turns xyxy boxes into xywh centers and keeps classes and scores."""
        import numpy as np

        from trocr_handwritten.parse.utils import DetectionResult

        r = DetectionResult(
            "p.jpg", np.array([[10, 20, 50, 60]]), np.array([3]), np.array([0.9])
        )
        assert r.path == "p.jpg"
        assert r.boxes.xywh.tolist() == [[30.0, 40.0, 40.0, 40.0]]
        assert r.boxes.cls.tolist() == [3.0]
        empty = DetectionResult("q.jpg", np.zeros((0, 4)), np.zeros(0), np.zeros(0))
        assert empty.boxes.xywh.shape == (0, 4)

    @patch("trocr_handwritten.parse.utils._load_model")
    def test_predict_uses_detect_and_feeds_predictions_json(self, mock_load):
        """predict() calls detect() per image; create_predictions_json reads the results."""
        import numpy as np

        from trocr_handwritten.parse.utils import create_predictions_json

        model = Mock(spec=["detect"])
        model.detect.return_value = (
            np.array([[100, 200, 300, 260]]),
            np.array([4]),
            np.array([0.8]),
        )
        mock_load.return_value = model
        settings = LayoutParserSettings(path_model="best.pt", conf=0.2, iou=0.5)
        parser = YOLOModel(settings, Mock())
        res = parser.predict(["a.jpg", "b.jpg"])
        assert model.detect.call_count == 2
        model.detect.assert_any_call("a.jpg", 0.2, 0.5, "cpu")
        out = create_predictions_json(res, CLASS_NAMES)
        assert out[0]["filename"] == "a.jpg"
        assert out[0]["regions"][0]["label"] == "Plein Texte"
        assert out[0]["regions"][0]["x"] == 100 and out[0]["regions"][0]["width"] == 200


class TestConfSweepHFAndBenchmark:
    """Tests for the confidence sweep, the Hugging Face layout conversion and the benchmark runner."""

    def test_conf_sweep_in_report(self, tmp_path):
        """A low-confidence false positive lowers F1 at conf 0.2 but not at 0.5."""
        import numpy as np
        from PIL import Image

        from trocr_handwritten.parse.metrics import evaluate

        for sub in ("images", "labels"):
            (tmp_path / "dev" / sub).mkdir(parents=True)
        Image.new("RGB", (100, 100), "white").save(
            tmp_path / "dev" / "images" / "p.jpg"
        )
        (tmp_path / "dev" / "labels" / "p.txt").write_text("4 0.3 0.3 0.2 0.2")
        model = Mock(spec=["detect"])
        model.detect.return_value = (
            np.array([[20.0, 20.0, 40.0, 40.0], [60.0, 60.0, 90.0, 90.0]]),
            np.array([4, 4]),
            np.array([0.9, 0.3]),
        )
        settings = EvaluationSettings(
            path_data=str(tmp_path), split="dev", conf_sweep=(0.2, 0.5)
        )
        report = evaluate(settings, Mock(), model=model, verbose=False)
        f1 = {c: report["conf_sweep"][c]["0.5"]["ALL"]["f1"] for c in ("0.2", "0.5")}
        assert round(f1["0.2"], 4) == round(2 / 3, 4)
        assert f1["0.5"] == 1.0
        assert report["prf"]["0.5"]["ALL"]["f1"] == f1["0.2"]

    def test_convert_layout_snapshot(self, tmp_path):
        """metadata.jsonl rows become annotations.json entries with class ids and image links."""
        import json

        from trocr_handwritten.parse.hf_layout import convert_layout_snapshot

        split_dir = tmp_path / "snap" / "layout" / "dev"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "images" / "p.jpg").write_bytes(b"jpg")
        row = {
            "file_name": "images/p.jpg",
            "source": "corpus",
            "register": "r1",
            "width": 10,
            "height": 20,
            "objects": {"bbox": [[1, 2, 3, 4]], "category": ["Signature"]},
        }
        (split_dir / "metadata.jsonl").write_text(json.dumps(row) + "\n")
        out = tmp_path / "out"
        assert convert_layout_snapshot(
            tmp_path / "snap", out, CLASS_NAMES, ("dev",)
        ) == {"dev": 1}
        entry = json.loads((out / "dev" / "annotations.json").read_text())[0]
        assert entry["boxes"] == [
            {
                "x": 1,
                "y": 2,
                "width": 3,
                "height": 4,
                "class_id": 5,
                "label": "Signature",
            }
        ]
        assert (out / "images_src" / "p.jpg").resolve() == (
            split_dir / "images" / "p.jpg"
        ).resolve()

    def test_benchmark_retrains_interrupted_run(self, tmp_path):
        """A best.pt without the marker is trained again; a finished run is skipped."""
        from trocr_handwritten.parse import benchmark
        from trocr_handwritten.parse.settings import BenchmarkSettings

        settings = BenchmarkSettings(path_data=str(tmp_path), images_dir=str(tmp_path))
        best = tmp_path / "models" / "bench_frcnn" / "weights" / "best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"partial")
        with patch.object(benchmark, "run_logged", return_value=0) as run:
            assert benchmark.train_run("bench_frcnn", settings)
            cmd = run.call_args[0][0]
            assert cmd[-2:] == ["--patience", "10"]
            assert (best.parent.parent / benchmark.DONE_MARKER).exists()
            run.reset_mock()
            assert benchmark.train_run("bench_frcnn", settings)
            run.assert_not_called()

    def test_benchmark_doclayout_env(self):
        """Only the DocLayout-YOLO run gets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD."""
        from trocr_handwritten.parse.benchmark import child_env
        from trocr_handwritten.parse.settings import BenchmarkSettings

        settings = BenchmarkSettings()
        assert (
            child_env("bench_doclayout", settings)["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]
            == "1"
        )
        import os

        assert child_env("bench_yolo26m", settings).get(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        ) == os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
