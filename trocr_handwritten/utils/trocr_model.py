from torch.utils.data import Dataset
import torch
from transformers import Seq2SeqTrainingArguments
from datasets import load_dataset
from os.path import join
import json
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_process_data(data_path, tokenizer, processor):
    """
    Load and process the training, validation, and test datasets.

    Args:
        data_path (str): The path to the data.
        tokenizer (Tokenizer): The tokenizer object to use for preprocessing.
        processor (Processor): The processor object to use for preprocessing.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The training, validation, and test datasets.
    """

    # Load the data
    raw_datasets = load_dataset(data_path)
    train_data, valid_data, test_data = (
        raw_datasets["train"],
        raw_datasets["validation"],
        raw_datasets["test"],
    )

    # Preprocess the data
    train_data = preprocess_dataset(train_data)
    valid_data = preprocess_dataset(valid_data)
    test_data = preprocess_dataset(test_data)

    # Split the data into training, validation, and test datasets
    train_dataset, valid_dataset, test_dataset = set_train_dev_test_split(
        train_data, valid_data, test_data, tokenizer, processor, max_length=64
    )

    # Log the number of examples in each split
    logging.info(f"Number of training examples: {len(train_dataset)}")
    logging.info(f"Number of validation examples: {len(valid_dataset)}")
    logging.info(f"Number of test examples: {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset


def preprocess_dataset(input_dataset):
    """
    Preprocess the dataset by replacing certain characters, removing extra spaces,
    and filtering out entries with no text.

    Args:
        input_dataset (Dataset): The dataset to preprocess.

    Returns:
        Dataset: The preprocessed dataset.
    """

    # Replace certain characters with spaces and strip leading/trailing spaces
    input_dataset = input_dataset.map(
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
    input_dataset = input_dataset.map(lambda x: {"text": " ".join(x["text"].split())})

    # Strip leading/trailing spaces again after replacing multiple spaces
    input_dataset = input_dataset.map(lambda x: {"text": x["text"].strip()})

    # Add a new field 'len' to the dataset which contains the length of the text
    input_dataset = input_dataset.map(lambda x: {"len": len(x["text"])})

    # Filter out entries where the text length is 0
    input_dataset = input_dataset.filter(lambda x: x["len"] > 0)

    return input_dataset


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


def set_train_dev_test_split(
    train_data,
    validation_data,
    test_data,
    text_tokenizer,
    image_processor,
    max_label_length=64,
):
    """
    Create training, validation, and test datasets.

    Args:
        train_data (Dataset): The training data.
        validation_data (Dataset): The validation data.
        test_data (Dataset): The test data.
        text_tokenizer (Tokenizer): The tokenizer to use for text tokenization.
        image_processor (Processor): The processor to use for image preprocessing.
        max_label_length (int): The maximum length of the tokenized text labels.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The training, validation, and test datasets.
    """

    train_dataset = OCRDataset(
        dataset=train_data,
        tokenizer=text_tokenizer,
        processor=image_processor,
        max_target_length=max_label_length,
    )

    eval_dataset = OCRDataset(
        dataset=validation_data,
        tokenizer=text_tokenizer,
        processor=image_processor,
        max_target_length=max_label_length,
    )

    test_dataset = OCRDataset(
        dataset=test_data,
        tokenizer=text_tokenizer,
        processor=image_processor,
        max_target_length=max_label_length,
    )

    return train_dataset, eval_dataset, test_dataset


def set_model_params(model, text_processor, config_path):
    """
    Configure the model parameters.

    Args:
        model (Model): The model to configure.
        text_processor (Processor): The processor to use for text processing.
        config_path (str): The path to the configuration file.

    Returns:
        Model: The configured model.
    """

    # Load the beam configuration
    with open(join(config_path, "beam_config.json")) as file:
        beam_config = json.load(file)

    # Set the special tokens used for creating the decoder input IDs from the labels
    model.config.decoder_start_token_id = text_processor.tokenizer.cls_token_id
    model.config.pad_token_id = text_processor.tokenizer.pad_token_id

    # Ensure the vocabulary size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set the beam search parameters
    model.config.eos_token_id = text_processor.tokenizer.sep_token_id
    model.config.max_length = beam_config.get("MAX_LENGTH", 64)
    model.config.early_stopping = beam_config.get("EARLY_STOPPING", True)
    model.config.no_repeat_ngram_size = beam_config.get("NO_REPEAT_NGRAM_SIZE", 3)
    model.config.length_penalty = beam_config.get("LENGTH_PENALTY", 2.0)
    model.config.num_beams = beam_config.get("NUM_BEAMS", 4)

    return model


def set_training_args(config_path):
    """
    Configure the training arguments.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Seq2SeqTrainingArguments: The configured training arguments.
    """

    # Load the training configuration
    with open(join(config_path, "training_config.json")) as file:
        training_config = json.load(file)

    # Configure the training arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=training_config.get("PREDICT_WITH_GENERATE", True),
        evaluation_strategy=training_config.get("EVALUATION_STRATEGY", "epoch"),
        save_strategy=training_config.get("SAVE_STRATEGY", "epoch"),
        per_device_train_batch_size=training_config.get(
            "PER_DEVICE_TRAIN_BATCH_SIZE", 16
        ),
        per_device_eval_batch_size=training_config.get(
            "PER_DEVICE_EVAL_BATCH_SIZE", 16
        ),
        gradient_accumulation_steps=training_config.get(
            "GRADIENT_ACCUMULATION_STEPS", 2
        ),
        optim=training_config.get("OPTIM", "adafactor"),
        num_train_epochs=training_config.get("NUM_TRAIN_EPOCHS", 10),
        fp16=training_config.get("FP16", True),
        learning_rate=training_config.get("LEARNING_RATE", 4e-5),
        output_dir=training_config.get("OUTPUT_DIR", "./results"),
        logging_dir=training_config.get("LOGGING_DIR", "./logs"),
        metric_for_best_model=training_config.get("METRIC_FOR_BEST_MODEL", "eval_loss"),
    )

    return training_args


def compute_metrics(cer_metric, wer_metric, text_tokenizer, text_processor):
    """
    Configure the metrics to use for evaluation.

    Args:
        character_error_rate_metric (Metric): The character error rate metric.
        word_error_rate_metric (Metric): The word error rate metric.
        text_tokenizer (Tokenizer): The tokenizer to use for text tokenization.
        text_processor (Processor): The processor to use for text processing.

    Returns:
        function: A function that computes the metrics.
    """

    def compute_predictions(predictions):
        """
        Compute the metrics for the given predictions.

        Args:
            predictions (Predictions): The predictions to compute the metrics for.

        Returns:
            dict: A dictionary containing the computed metrics.
        """

        # Get the label IDs and prediction IDs
        label_ids = predictions.label_ids
        prediction_ids = predictions.predictions

        # Decode the prediction IDs and label IDs
        prediction_strings = text_tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )
        label_ids[label_ids == -100] = text_processor.tokenizer.pad_token_id
        label_strings = text_tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute the character error rate and word error rate
        character_error_rate = cer_metric.compute(
            predictions=prediction_strings, references=label_strings
        )
        word_error_rate = wer_metric.compute(
            predictions=prediction_strings, references=label_strings
        )

        return {"cer": character_error_rate, "wer": word_error_rate}

    return compute_predictions
