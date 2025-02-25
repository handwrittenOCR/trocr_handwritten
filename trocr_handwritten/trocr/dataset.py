import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import AutoTokenizer, TrOCRProcessor
from datasets import load_dataset
import logging
from collections import Counter
import random
from typing import Dict, Tuple
from huggingface_hub import login
from trocr_handwritten.trocr.settings import TrainerDatasetsSettings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OCRDataset(Dataset):
    """
    A PyTorch Dataset for Optical Character Recognition (OCR).
    """

    def __init__(self, data, image_processor, text_tokenizer, max_label_length=32):
        """
        Initialize the dataset with data, an image processor, a text tokenizer, and a maximum label length.

        Args:
            data (Dataset): The dataset to use.
            image_processor (Processor): The processor to use for image preprocessing.
            text_tokenizer (Tokenizer): The tokenizer to use for text tokenization.
            max_label_length (int): The maximum length of the tokenized text labels.
        """
        self.data = data
        self.image_processor = image_processor
        self.max_label_length = max_label_length
        self.text_tokenizer = text_tokenizer

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
        # Get the item and extract the image and text
        item = self.data[index]
        image = item["image"].convert("RGB")
        text = item["text"]

        # Preprocess the image
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values

        # Tokenize the text and get the input IDs as labels
        labels = self.text_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_label_length,
        ).input_ids

        # Replace padding token IDs with -100 so they are ignored by the loss function
        labels = [
            label if label != self.text_tokenizer.pad_token_id else -100
            for label in labels
        ]

        # Return the preprocessed image and labels
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}


class TrainerDatasets:
    """
    A class to handle loading, preprocessing, and creating datasets for training.
    """

    def __init__(
        self,
        settings: TrainerDatasetsSettings,
        tokenizer: AutoTokenizer,
        processor: TrOCRProcessor,
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
        self.private_repo = settings.private_repo
        self.max_label_length = settings.max_label_length
        self.huggingface_api_key = settings.huggingface_api_key
        self.datasets = {}

    def load_and_process_data(self) -> Dict[str, Dataset]:
        """
        Load and process the datasets.

        Returns:
            Dict[str, Dataset]: A dictionary containing the training, validation, and test datasets.
        """
        all_datasets = {}

        # Load census data if requested
        if self.census_data:
            logger.info("Loading census dataset...")
            census_datasets = self._load_census_data()
            census_datasets = self._ensure_all_splits_exist(census_datasets)
            all_datasets["census"] = census_datasets

        # Load private repository data if provided
        if self.private_repo:
            logger.info(f"Loading private dataset from {self.private_repo}...")
            private_datasets = self._load_private_data()
            private_datasets = self._ensure_all_splits_exist(private_datasets)
            all_datasets["private"] = private_datasets

        # Combine datasets if both are present
        if self.census_data and self.private_repo:
            logger.info("Combining census and private datasets...")
            combined_datasets = self._combine_datasets(
                all_datasets["census"], all_datasets["private"]
            )
            self.datasets = combined_datasets
        elif self.census_data:
            self.datasets = all_datasets["census"]
        elif self.private_repo:
            self.datasets = all_datasets["private"]
        else:
            raise ValueError(
                "No datasets specified. Set census_data=True or provide private_repo."
            )

        logger.info(
            f"Final dataset sizes - Train: {len(self.datasets['train'])}, "
            f"Validation: {len(self.datasets['validation'])}, "
            f"Test: {len(self.datasets['test'])}"
        )

        return self.datasets

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
        split = int(test_size * dataset_size)
        random.shuffle(indices)

        train_indices, test_indices = indices[split:], indices[:split]

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
        # Check if all required splits exist
        required_splits = ["train", "validation", "test"]
        missing_splits = [split for split in required_splits if split not in datasets]

        if not missing_splits:
            return datasets  # All splits exist

        if "train" not in datasets:
            raise ValueError("Training dataset is required but missing.")

        # Create missing splits from the training data
        train_dataset = datasets["train"]

        for split in missing_splits:
            logger.warning(
                f"{split.capitalize()} split is missing. Creating from training data."
            )

            if split == "validation":
                # Use 10% of training data for validation
                train_dataset, val_dataset = self._split_dataset(
                    train_dataset, test_size=0.1
                )
                datasets["train"] = train_dataset
                datasets["validation"] = val_dataset

            elif split == "test":
                # Use 10% of training data for testing
                train_dataset, test_dataset = self._split_dataset(
                    train_dataset, test_size=0.1
                )
                datasets["train"] = train_dataset
                datasets["test"] = test_dataset

        return datasets

    def _load_census_data(self) -> Dict[str, Dataset]:
        """
        Load and process the census dataset.

        Returns:
            Dict[str, Dataset]: A dictionary containing the training, validation, and test datasets.
        """
        # Load the census dataset
        try:
            raw_datasets = load_dataset("agomberto/FrenchCensus-handwritten-texts")

            # Check which splits are available
            available_splits = raw_datasets.keys()
            datasets = {}

            # Process available splits
            if "train" in available_splits:
                train_data = self._preprocess_census_data(raw_datasets["train"])
                datasets["train"] = OCRDataset(
                    data=train_data,
                    image_processor=self.processor,
                    text_tokenizer=self.tokenizer,
                    max_label_length=self.max_label_length,
                )

            if "validation" in available_splits:
                valid_data = self._preprocess_census_data(raw_datasets["validation"])
                datasets["validation"] = OCRDataset(
                    data=valid_data,
                    image_processor=self.processor,
                    text_tokenizer=self.tokenizer,
                    max_label_length=self.max_label_length,
                )

            if "test" in available_splits:
                test_data = self._preprocess_census_data(raw_datasets["test"])
                datasets["test"] = OCRDataset(
                    data=test_data,
                    image_processor=self.processor,
                    text_tokenizer=self.tokenizer,
                    max_label_length=self.max_label_length,
                )

            return datasets

        except Exception as e:
            logger.error(f"Error loading census dataset: {str(e)}")
            raise

    def _preprocess_census_data(self, dataset):
        """
        Preprocess the census dataset by replacing certain characters, removing extra spaces,
        and filtering out entries with no text.

        Args:
            dataset (Dataset): The dataset to preprocess.

        Returns:
            Dataset: The preprocessed dataset.
        """
        # Replace certain characters with spaces and strip leading/trailing spaces
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

        # Replace multiple spaces with a single space
        dataset = dataset.map(lambda x: {"text": " ".join(x["text"].split())})

        # Strip leading/trailing spaces again after replacing multiple spaces
        dataset = dataset.map(lambda x: {"text": x["text"].strip()})

        # Add a new field 'len' to the dataset which contains the length of the text
        dataset = dataset.map(lambda x: {"len": len(x["text"])})

        # Filter out entries where the text length is 2
        dataset = dataset.filter(lambda x: x["len"] > 2)

        return dataset

    def _load_private_data(self) -> Dict[str, Dataset]:
        """
        Load and process the private dataset.

        Returns:
            Dict[str, Dataset]: A dictionary containing the training, validation, and test datasets.
        """
        # Load the private dataset
        try:
            # Login to Hugging Face Hub if API key is provided
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

            # Check which splits are available
            available_splits = raw_datasets.keys()
            datasets = {}

            # Process available splits
            if "train" in available_splits:
                train_data = self._preprocess_private_dataset(raw_datasets["train"])
                datasets["train"] = OCRDataset(
                    data=train_data,
                    image_processor=self.processor,
                    text_tokenizer=self.tokenizer,
                    max_label_length=self.max_label_length,
                )

            if "validation" in available_splits:
                valid_data = self._preprocess_private_dataset(
                    raw_datasets["validation"]
                )
                datasets["validation"] = OCRDataset(
                    data=valid_data,
                    image_processor=self.processor,
                    text_tokenizer=self.tokenizer,
                    max_label_length=self.max_label_length,
                )

            if "test" in available_splits:
                test_data = self._preprocess_private_dataset(raw_datasets["test"])
                datasets["test"] = OCRDataset(
                    data=test_data,
                    image_processor=self.processor,
                    text_tokenizer=self.tokenizer,
                    max_label_length=self.max_label_length,
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
        # Filter out entries with text length less than 3
        dataset = dataset.filter(lambda x: len(x["text"]) >= length_threshold)

        # Limit the number of occurrences of each text to 0.05% of the dataset
        text_counter = Counter(dataset["text"])
        max_occurrences = max(1, int(len(dataset) * max_occurrences))  # 0.05%

        # Create a list of indices to keep
        indices_to_keep = []
        text_count = {text: 0 for text in text_counter.keys()}

        for idx, text in enumerate(dataset["text"]):
            if text_count[text] < max_occurrences:
                indices_to_keep.append(idx)
                text_count[text] += 1

        # Shuffle the indices to ensure randomness
        random.shuffle(indices_to_keep)

        # Select the filtered dataset
        filtered_dataset = dataset.select(indices_to_keep)

        logger.info(
            f"Private dataset filtered from {len(dataset)} to {len(filtered_dataset)} examples"
        )

        return filtered_dataset

    def _combine_datasets(
        self, census_datasets: Dict[str, Dataset], private_datasets: Dict[str, Dataset]
    ) -> Dict[str, Dataset]:
        """
        Combine the census and private datasets.

        Args:
            census_datasets (Dict[str, Dataset]): The census datasets.
            private_datasets (Dict[str, Dataset]): The private datasets.

        Returns:
            Dict[str, Dataset]: The combined datasets.
        """
        combined_datasets = {}

        # Get all available splits from both datasets
        all_splits = set(list(census_datasets.keys()) + list(private_datasets.keys()))

        for split in all_splits:
            datasets_to_combine = []

            if split in census_datasets:
                datasets_to_combine.append(census_datasets[split])

            if split in private_datasets:
                datasets_to_combine.append(private_datasets[split])

            if datasets_to_combine:
                combined_datasets[split] = ConcatDataset(datasets_to_combine)

        return combined_datasets
