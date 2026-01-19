import pytest
from unittest.mock import Mock, patch

from trocr_handwritten.parse.settings import LayoutParserSettings
from trocr_handwritten.parse.utils import calculate_iou, YOLOv10Model


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
        """Test that class_names are initialized in __post_init__."""
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


class TestYOLOv10Model:
    """Tests for the YOLOv10Model class."""

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

    @patch("trocr_handwritten.parse.utils.hf_hub_download")
    @patch("trocr_handwritten.parse.utils.YOLOv10")
    def test_load_from_hf(self, mock_yolo, mock_download, mock_settings, mock_logger):
        """Test loading model from HuggingFace Hub."""
        mock_download.return_value = "/path/to/model.pt"
        mock_yolo.return_value = Mock()

        _ = YOLOv10Model(mock_settings, mock_logger)

        mock_download.assert_called_once_with(
            repo_id="test/repo",
            filename="model.pt",
        )
        mock_yolo.assert_called_once_with("/path/to/model.pt")

    @patch("trocr_handwritten.parse.utils.YOLOv10")
    def test_load_from_local(self, mock_yolo, mock_logger):
        """Test loading model from local path."""
        settings = LayoutParserSettings(
            path_model="/local/model.pt",
            hf_repo=None,
        )
        mock_yolo.return_value = Mock()

        _ = YOLOv10Model(settings, mock_logger)

        mock_yolo.assert_called_once_with("/local/model.pt")

    def test_raises_without_model_source(self, mock_logger):
        """Test that error is raised when no model source provided."""
        settings = LayoutParserSettings(
            path_model=None,
            hf_repo=None,
            hf_filename=None,
        )

        with pytest.raises(ValueError, match="Either path_model or hf_repo"):
            YOLOv10Model(settings, mock_logger)

    @patch("trocr_handwritten.parse.utils.hf_hub_download")
    @patch("trocr_handwritten.parse.utils.YOLOv10")
    def test_predict_calls_model(
        self, mock_yolo, mock_download, mock_settings, mock_logger
    ):
        """Test that predict calls the underlying model."""
        mock_download.return_value = "/path/to/model.pt"
        mock_model_instance = Mock()
        mock_yolo.return_value = mock_model_instance

        model = YOLOv10Model(mock_settings, mock_logger)
        model.predict("/test/folder")

        mock_model_instance.predict.assert_called_once_with(
            "/test/folder",
            imgsz=1024,
            conf=mock_settings.conf,
            iou=mock_settings.iou,
            device=mock_settings.device,
        )
