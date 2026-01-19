import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import AutoTokenizer, TrOCRProcessor
from datasets import load_dataset
from collections import Counter
import random
import numpy as np
from typing import Dict, Tuple, Callable, Optional
from huggingface_hub import login
from trocr_handwritten.trocr.settings import TrainerDatasetsSettings
from trocr_handwritten.utils.logging_config import get_logger
import cv2
from PIL import Image

logger = get_logger(__name__)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def apply_gray(img):
    """Convert to grayscale."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def apply_denoise(img, h=5, templateWindowSize=5, searchWindowSize=21):
    """Apply denoising to image."""
    return cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)


class OCRDataset(Dataset):
    """A PyTorch Dataset for Optical Character Recognition (OCR)."""

    def __init__(
        self,
        data,
        image_processor,
        text_tokenizer,
        max_label_length=32,
        dataset_source=None,
    ):
        """
        Initialize the dataset with data, an image processor, a text tokenizer, and a maximum label length.

        Args:
            data (Dataset): The dataset to use.
            image_processor (Processor): The processor to use for image preprocessing.
            text_tokenizer (Tokenizer): The tokenizer to use for text tokenization.
            max_label_length (int): The maximum length of the tokenized text labels.
            dataset_source (str): The source of the dataset (e.g., 'census', 'private').
        """
        self.data = data
        self.image_processor = image_processor
        self.max_label_length = max_label_length
        self.text_tokenizer = text_tokenizer
        self.dataset_source = dataset_source

    def __len__(self):
        """
        Get the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Get the item at the specified index.

        Args:
            index (int): The index of the item.

        Returns:
            dict: A dictionary containing the preprocessed image and the tokenized text label.
        """
        item = self.data[index]
        image = item["image"].convert("RGB")
        text = item["text"]

        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values

        labels = self.text_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_label_length,
        ).input_ids

        labels = [
            label if label != self.text_tokenizer.pad_token_id else -100
            for label in labels
        ]

        result = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "dataset_source": self.dataset_source,
        }
        return result


class TrainerDatasets:
    """A class to handle loading, preprocessing, and creating datasets for training."""

    DATASET_CONFIGS = {
        "rimes": {
            "name": "Teklia/RIMES-2011-line",
            "source": "rimes",
            "preprocess_fn": None,
        },
        "belfort": {
            "name": "Teklia/Belfort-line",
            "source": "belfort",
            "preprocess_fn": None,
        },
        "census": {
            "name": "agomberto/FrenchCensus-handwritten-texts",
            "source": "census",
            "preprocess_fn": "_preprocess_census_data",
        },
    }

    def __init__(
        self,
        settings: TrainerDatasetsSettings,
        tokenizer: AutoTokenizer,
        processor: TrOCRProcessor,
        preprocess_images: bool = True,
        evaluate_only: bool = False,
    ):
        """
        Initialize the TrainerDatasets class.

        Args:
            settings: TrainerDatasetsSettings instance containing dataset configuration.
        """
        self.settings = settings
        self.tokenizer = tokenizer
        self.processor = processor
        self.census_data = settings.census_data
        self.rimes_data = settings.rimes_data
        self.belfort_data = settings.belfort_data
        self.private_repo = settings.private_repo
        self.max_label_length = settings.max_label_length
        self.huggingface_api_key = settings.huggingface_api_key
        self.preprocess_images = settings.preprocess_images
        self.datasets = {}
        self.evaluate_only = evaluate_only

        self.seed = RANDOM_SEED

    def load_and_process_data(self) -> Dict[str, Dataset]:
        """
        Load and process the datasets.

        Returns:
            Dict[str, Dataset]: A dictionary containing the training, validation, and test datasets.
        """
        all_datasets = {}

        dataset_flags = {
            "rimes": self.rimes_data,
            "belfort": self.belfort_data,
            "census": self.census_data,
        }

        for dataset_key, is_enabled in dataset_flags.items():
            if is_enabled:
                logger.info(f"Loading {dataset_key} dataset...")
                config = self.DATASET_CONFIGS[dataset_key]
                preprocess_fn = None
                if config["preprocess_fn"]:
                    preprocess_fn = getattr(self, config["preprocess_fn"], None)
                loaded = self._load_dataset(
                    dataset_name=config["name"],
                    source=config["source"],
                    preprocess_fn=preprocess_fn,
                )
                loaded = self._ensure_all_splits_exist(loaded)
                all_datasets[dataset_key] = loaded

        if self.private_repo:
            logger.info(f"Loading private dataset from {self.private_repo}...")
            private_datasets = self._load_private_data()
            private_datasets = self._ensure_all_splits_exist(private_datasets)
            all_datasets["private"] = private_datasets

        args = [all_datasets[key] for key in all_datasets.keys()]
        logger.info("Combining downloaded datasets...")
        combined_datasets = self._combine_datasets(*args)
        self.datasets = combined_datasets

        logger.info(
            f"Final dataset sizes - Train: {len(self.datasets['train'])}, "
            f"Validation: {len(self.datasets['validation'])}, "
            f"Test: {len(self.datasets['test'])}"
        )

        return self.datasets

    def _load_dataset(
        self,
        dataset_name: str,
        source: str,
        preprocess_fn: Optional[Callable] = None,
    ) -> Dict[str, Dataset]:
        """
        Generic method to load and process a dataset from HuggingFace.

        Args:
            dataset_name: HuggingFace dataset identifier.
            source: Source name for tracking.
            preprocess_fn: Optional preprocessing function to apply.

        Returns:
            Dictionary with train/validation/test OCRDataset objects.
        """
        try:
            raw_datasets = load_dataset(dataset_name)
            available_splits = raw_datasets.keys()
            datasets = {}

            for split in ["train", "validation", "test"]:
                if split not in available_splits:
                    continue

                data = raw_datasets[split]

                if preprocess_fn:
                    data = preprocess_fn(data)

                should_preprocess = split == "test" or not self.evaluate_only
                if should_preprocess:
                    data = self._preprocess_images(data)

                datasets[split] = OCRDataset(
                    data=data,
                    image_processor=self.processor,
                    text_tokenizer=self.tokenizer,
                    max_label_length=self.max_label_length,
                    dataset_source=source,
                )

            return datasets

        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {str(e)}")
            raise

    def _split_dataset(self, dataset, test_size: float) -> Tuple[Dataset, Dataset]:
        """
        Split a dataset into training and validation/test sets.

        Args:
            dataset (Dataset): The dataset to split.
            test_size (float): The proportion of the dataset to include in the test split.

        Returns:
            Tuple[Dataset, Dataset]: The training and validation/test datasets.
        """
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)

        split = int(test_size * dataset_size)

        train_indices, test_indices = indices[split:], indices[:split]

        train_indices.sort()
        test_indices.sort()

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)

        return train_subset, test_subset

    def _ensure_all_splits_exist(
        self, datasets: Dict[str, Dataset]
    ) -> Dict[str, Dataset]:
        """
        Ensure that training, validation, and test splits exist.
        If validation or test splits are missing, create them from the training data.

        Args:
            datasets (Dict[str, Dataset]): The datasets to check.

        Returns:
            Dict[str, Dataset]: The datasets with all required splits.
        """
        required_splits = ["train", "validation", "test"]
        missing_splits = [split for split in required_splits if split not in datasets]

        if not missing_splits:
            return datasets

        if "train" not in datasets:
            raise ValueError("Training dataset is required but missing.")

        train_dataset = datasets["train"]

        logger.info(
            "Creating missing splits with fixed random seed for reproducibility"
        )

        if "validation" in missing_splits and "test" in missing_splits:
            train_dataset, temp_dataset = self._split_dataset(
                train_dataset, test_size=0.2
            )

            validation_dataset, test_dataset = self._split_dataset(
                temp_dataset, test_size=0.5
            )

            datasets["train"] = train_dataset
            datasets["validation"] = validation_dataset
            datasets["test"] = test_dataset

            logger.info(
                "Created validation and test splits (10% each) from training data"
            )

        elif "validation" in missing_splits:
            train_dataset, validation_dataset = self._split_dataset(
                train_dataset, test_size=0.1
            )
            datasets["train"] = train_dataset
            datasets["validation"] = validation_dataset
            logger.info("Created validation split (10%) from training data")

        elif "test" in missing_splits:
            train_dataset, test_dataset = self._split_dataset(
                train_dataset, test_size=0.1
            )
            datasets["train"] = train_dataset
            datasets["test"] = test_dataset
            logger.info("Created test split (10%) from training data")

        return datasets

    def _preprocess_census_data(self, dataset):
        """
        Preprocess the census dataset by replacing certain characters, removing extra spaces,
        and filtering out entries with no text.

        Args:
            dataset (Dataset): The dataset to preprocess.

        Returns:
            Dataset: The preprocessed dataset.
        """
        dataset = dataset.map(
            lambda x: {
                "text": x["text"]
                .replace("¤", " ")
                .replace("/", " ")
                .replace("�", " ")
                .replace("¬", " ")
                .strip()
            }
        )

        dataset = dataset.map(lambda x: {"text": " ".join(x["text"].split())})

        dataset = dataset.map(lambda x: {"text": x["text"].strip()})

        dataset = dataset.map(lambda x: {"len": len(x["text"])})

        dataset = dataset.filter(lambda x: x["len"] > 2)

        dataset = dataset.map(lambda x: {"source_image": ""})

        return dataset

    def _preprocess_images(self, dataset):
        """
        Preprocess images in the dataset using map.

        Args:
            dataset (Dataset): The dataset to preprocess.

        Returns:
            Dataset: The dataset with preprocessed images.
        """

        def preprocess_image(item):
            if self.preprocess_images:
                image = item["image"].convert("RGB")
                image = apply_gray(np.array(image))
                image = apply_denoise(image)
                item["image"] = Image.fromarray(image).convert("RGB")
            return item

        return dataset.map(preprocess_image)

    def _load_private_data(self) -> Dict[str, Dataset]:
        """
        Load and process the private dataset.

        Returns:
            Dict[str, Dataset]: A dictionary containing the training, validation, and test datasets.
        """
        try:
            if self.huggingface_api_key:
                try:
                    login(token=self.huggingface_api_key)
                    logger.info("Successfully logged in to Hugging Face Hub")
                except ImportError:
                    logger.warning(
                        "huggingface_hub not installed. Cannot login with API key."
                    )
                except Exception as e:
                    logger.warning(f"Failed to login to Hugging Face Hub: {str(e)}")

            raw_datasets = load_dataset(self.private_repo)
            available_splits = raw_datasets.keys()
            datasets = {}

            for split in ["train", "validation", "test"]:
                if split not in available_splits:
                    continue

                data = self._preprocess_private_dataset(raw_datasets[split])

                should_preprocess = split == "test" or not self.evaluate_only
                if should_preprocess:
                    data = self._preprocess_images(data)

                datasets[split] = OCRDataset(
                    data=data,
                    image_processor=self.processor,
                    text_tokenizer=self.tokenizer,
                    max_label_length=self.max_label_length,
                    dataset_source="private",
                )

            return datasets

        except Exception as e:
            logger.error(f"Error loading private dataset: {str(e)}")
            raise

    def _preprocess_private_dataset(
        self, dataset, length_threshold=3, max_occurrences=0.0005
    ):
        """
        Apply custom preprocessing to the private dataset.

        Args:
            dataset (Dataset): The dataset to preprocess.
            length_threshold (int): The minimum length of the text to keep.
        Returns:
            Dataset: The preprocessed dataset.
        """
        dataset = dataset.filter(lambda x: len(x["text"]) >= length_threshold)

        text_counter = Counter(dataset["text"])
        max_occurrences = max(1, int(len(dataset) * max_occurrences))

        indices_to_keep = []
        text_count = {text: 0 for text in text_counter.keys()}

        for idx, text in enumerate(dataset["text"]):
            if text_count[text] < max_occurrences:
                indices_to_keep.append(idx)
                text_count[text] += 1

        random.shuffle(indices_to_keep)

        filtered_dataset = dataset.select(indices_to_keep)

        logger.info(
            f"Private dataset filtered from {len(dataset)} to {len(filtered_dataset)} examples"
        )

        return filtered_dataset

    def _combine_datasets(self, *args: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """
        Combine the census and private datasets.

        Args:
            census_datasets (Dict[str, Dataset]): The census datasets.
            private_datasets (Dict[str, Dataset]): The private datasets.

        Returns:
            Dict[str, Dataset]: The combined datasets.
        """
        combined_datasets = {}

        all_splits = set()
        for arg in args:
            all_splits.update(arg.keys())

        for split in all_splits:
            datasets_to_combine = []
            for dataset in args:
                if split in dataset:
                    datasets_to_combine.append(dataset[split])

            if datasets_to_combine:
                combined_datasets[split] = ConcatDataset(datasets_to_combine)

        return combined_datasets
