import pytest
import torch
import numpy as np
from unittest.mock import Mock
from PIL import Image

from trocr_handwritten.trocr.dataset import (
    OCRDataset,
    TrainerDatasets,
    apply_gray,
    RANDOM_SEED,
)
from trocr_handwritten.trocr.settings import TrainerDatasetsSettings


class TestApplyGray:
    """Tests for the apply_gray function."""

    def test_rgb_to_gray(self):
        """Test conversion of RGB image to grayscale."""
        rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_img[:, :, 0] = 255
        result = apply_gray(rgb_img)
        assert len(result.shape) == 2
        assert result.shape == (100, 100)

    def test_already_gray(self):
        """Test that grayscale image is returned unchanged."""
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        result = apply_gray(gray_img)
        assert len(result.shape) == 2
        assert np.array_equal(result, gray_img)


class TestOCRDataset:
    """Tests for the OCRDataset class."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock image processor."""
        processor = Mock()
        processor.return_value.pixel_values = torch.zeros(1, 3, 384, 384)
        return processor

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value.input_ids = [1, 2, 3, 0, 0, 0]
        tokenizer.pad_token_id = 0
        return tokenizer

    @pytest.fixture
    def mock_data(self):
        """Create mock dataset data."""
        img = Image.new("RGB", (100, 100), color="white")
        return [
            {"image": img, "text": "test text 1"},
            {"image": img, "text": "test text 2"},
        ]

    def test_len(self, mock_data, mock_processor, mock_tokenizer):
        """Test __len__ returns correct size."""
        dataset = OCRDataset(
            data=mock_data,
            image_processor=mock_processor,
            text_tokenizer=mock_tokenizer,
            max_label_length=32,
            dataset_source="test",
        )
        assert len(dataset) == 2

    def test_getitem_returns_correct_keys(
        self, mock_data, mock_processor, mock_tokenizer
    ):
        """Test __getitem__ returns expected dictionary keys."""
        dataset = OCRDataset(
            data=mock_data,
            image_processor=mock_processor,
            text_tokenizer=mock_tokenizer,
            max_label_length=32,
            dataset_source="test",
        )
        item = dataset[0]
        assert "pixel_values" in item
        assert "labels" in item
        assert "dataset_source" in item

    def test_getitem_dataset_source(self, mock_data, mock_processor, mock_tokenizer):
        """Test that dataset_source is correctly set."""
        dataset = OCRDataset(
            data=mock_data,
            image_processor=mock_processor,
            text_tokenizer=mock_tokenizer,
            max_label_length=32,
            dataset_source="census",
        )
        item = dataset[0]
        assert item["dataset_source"] == "census"

    def test_labels_padding_replaced(self, mock_data, mock_processor, mock_tokenizer):
        """Test that padding tokens are replaced with -100."""
        dataset = OCRDataset(
            data=mock_data,
            image_processor=mock_processor,
            text_tokenizer=mock_tokenizer,
            max_label_length=32,
            dataset_source="test",
        )
        item = dataset[0]
        labels = item["labels"].tolist()
        assert -100 in labels


class TestTrainerDatasets:
    """Tests for the TrainerDatasets class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = TrainerDatasetsSettings(
            census_data=False,
            rimes_data=False,
            belfort_data=False,
            private_repo=None,
            max_label_length=64,
            huggingface_api_key=None,
            preprocess_images=False,
        )
        return settings

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        return tokenizer

    @pytest.fixture
    def mock_processor(self):
        """Create mock processor."""
        return Mock()

    def test_init(self, mock_settings, mock_tokenizer, mock_processor):
        """Test TrainerDatasets initialization."""
        trainer = TrainerDatasets(
            settings=mock_settings,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )
        assert trainer.seed == RANDOM_SEED
        assert trainer.datasets == {}
        assert trainer.evaluate_only is False

    def test_dataset_configs_structure(self):
        """Test that DATASET_CONFIGS has expected structure."""
        configs = TrainerDatasets.DATASET_CONFIGS

        assert "rimes" in configs
        assert "belfort" in configs
        assert "census" in configs

        for key, config in configs.items():
            assert "name" in config
            assert "source" in config
            assert "preprocess_fn" in config

    def test_split_dataset(self, mock_settings, mock_tokenizer, mock_processor):
        """Test _split_dataset creates correct proportions."""
        trainer = TrainerDatasets(
            settings=mock_settings,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        train, test = trainer._split_dataset(mock_dataset, test_size=0.2)

        assert len(train.indices) == 80
        assert len(test.indices) == 20

    def test_split_dataset_reproducibility(
        self, mock_settings, mock_tokenizer, mock_processor
    ):
        """Test that splits are reproducible with same seed."""
        trainer1 = TrainerDatasets(
            settings=mock_settings,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )
        trainer2 = TrainerDatasets(
            settings=mock_settings,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        train1, _ = trainer1._split_dataset(mock_dataset, test_size=0.2)
        train2, _ = trainer2._split_dataset(mock_dataset, test_size=0.2)

        assert train1.indices == train2.indices

    def test_ensure_all_splits_exist_no_missing(
        self, mock_settings, mock_tokenizer, mock_processor
    ):
        """Test _ensure_all_splits_exist when all splits present."""
        trainer = TrainerDatasets(
            settings=mock_settings,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        datasets = {
            "train": Mock(),
            "validation": Mock(),
            "test": Mock(),
        }

        result = trainer._ensure_all_splits_exist(datasets)
        assert result == datasets

    def test_ensure_all_splits_raises_without_train(
        self, mock_settings, mock_tokenizer, mock_processor
    ):
        """Test that missing train split raises error."""
        trainer = TrainerDatasets(
            settings=mock_settings,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        datasets = {"validation": Mock()}

        with pytest.raises(ValueError, match="Training dataset is required"):
            trainer._ensure_all_splits_exist(datasets)

    def test_combine_datasets_single(
        self, mock_settings, mock_tokenizer, mock_processor
    ):
        """Test _combine_datasets with single dataset."""
        trainer = TrainerDatasets(
            settings=mock_settings,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        mock_train = Mock()
        mock_train.__len__ = Mock(return_value=100)
        mock_val = Mock()
        mock_val.__len__ = Mock(return_value=20)
        mock_test = Mock()
        mock_test.__len__ = Mock(return_value=20)

        datasets = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        result = trainer._combine_datasets(datasets)

        assert "train" in result
        assert "validation" in result
        assert "test" in result

    def test_combine_datasets_multiple(
        self, mock_settings, mock_tokenizer, mock_processor
    ):
        """Test _combine_datasets with multiple datasets."""
        trainer = TrainerDatasets(
            settings=mock_settings,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        mock_train1 = Mock()
        mock_train1.__len__ = Mock(return_value=100)
        mock_test1 = Mock()
        mock_test1.__len__ = Mock(return_value=20)
        mock_train2 = Mock()
        mock_train2.__len__ = Mock(return_value=100)
        mock_val2 = Mock()
        mock_val2.__len__ = Mock(return_value=20)

        datasets1 = {"train": mock_train1, "test": mock_test1}
        datasets2 = {"train": mock_train2, "validation": mock_val2}

        result = trainer._combine_datasets(datasets1, datasets2)

        assert "train" in result
        assert "validation" in result
        assert "test" in result


class TestPreprocessCensusData:
    """Tests for census data preprocessing."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        return TrainerDatasetsSettings(
            census_data=False,
            rimes_data=False,
            belfort_data=False,
            private_repo=None,
            preprocess_images=False,
        )

    @pytest.fixture
    def trainer(self, mock_settings):
        """Create trainer instance."""
        return TrainerDatasets(
            settings=mock_settings,
            tokenizer=Mock(),
            processor=Mock(),
        )

    def test_preprocess_census_removes_special_chars(self, trainer):
        """Test that special characters are removed."""
        mock_dataset = Mock()

        def capture_map(fn):
            result = Mock()
            result.map = capture_map
            result.filter = lambda f: result
            return result

        mock_dataset.map = capture_map
        mock_dataset.filter = lambda f: mock_dataset

        trainer._preprocess_census_data(mock_dataset)
