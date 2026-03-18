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
        assert settings.epochs == 50
        assert settings.imgsz == 1024
        assert settings.batch == 8
        assert settings.class_names == CLASS_NAMES

    def test_custom_values(self):
        """Test custom training settings."""
        settings = TrainingSettings(epochs=100, batch=16, model_base="yolo11s.pt")
        assert settings.epochs == 100
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

    @patch("trocr_handwritten.parse.utils._detect_model_backend")
    @patch("trocr_handwritten.parse.utils.hf_hub_download")
    @patch("trocr_handwritten.parse.utils._load_model")
    def test_load_from_hf(
        self, mock_load, mock_download, mock_detect, mock_settings, mock_logger
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

    @patch("trocr_handwritten.parse.utils._detect_model_backend")
    @patch("trocr_handwritten.parse.utils.hf_hub_download")
    @patch("trocr_handwritten.parse.utils._load_model")
    def test_predict_calls_model(
        self, mock_load, mock_download, mock_detect, mock_settings, mock_logger
    ):
        """Test that predict calls the underlying model."""
        mock_download.return_value = "/path/to/model.pt"
        mock_detect.return_value = "ultralytics"
        mock_model_instance = Mock()
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
