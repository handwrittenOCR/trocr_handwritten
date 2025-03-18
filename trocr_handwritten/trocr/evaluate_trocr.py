import logging
import argparse
import os
from trocr_handwritten.trocr.dataset import TrainerDatasets
from trocr_handwritten.trocr.model import OCRModel
from trocr_handwritten.trocr.settings import (
    OCRModelSettings,
    TrainerDatasetsSettings,
    TrainSettings,
)
from huggingface_hub import login
import dotenv

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_name: str,
    hf_token: str = None,
    census_data: bool = True,
    private_repo: str = None,
    max_label_length: int = 64,
) -> dict:
    """
    Evaluate a model on test datasets.

    Args:
        model_name: Name or path of the model on HuggingFace Hub
        hf_token: HuggingFace token for private repositories
        census_data: Whether to use census data
        private_repo: Private dataset repository
        max_label_length: Maximum label length

    Returns:
        dict: Evaluation metrics
    """
    # Login to HuggingFace if token provided
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if hf_token:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace Hub")

    # Configure settings
    model_settings = OCRModelSettings(
        model_name=model_name,
        processor_name=model_name,
        hub_repo=model_name,
        huggingface_api_key=hf_token,
    )

    train_settings = TrainSettings()

    dataset_settings = TrainerDatasetsSettings(
        census_data=census_data,
        private_repo=private_repo,
        max_label_length=max_label_length,
        huggingface_api_key=hf_token,
    )

    # Initialize model
    logger.info(f"Loading model from {model_name}")
    ocr_model = OCRModel(settings=model_settings, train_settings=train_settings)

    # Initialize datasets
    logger.info("Loading datasets")
    trainer_datasets = TrainerDatasets(
        settings=dataset_settings,
        tokenizer=ocr_model.tokenizer,
        processor=ocr_model.processor,
    )

    # Load and process data
    datasets = trainer_datasets.load_and_process_data()

    # Set up metrics computation
    compute_metrics_fn = OCRModel.setup_compute_metrics(
        ocr_model.tokenizer, ocr_model.processor
    )

    # Evaluate on test set
    logger.info("Evaluating model on test set")
    test_metrics = ocr_model.evaluate(
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        test_dataset=datasets["test"],
        compute_metrics_fn=compute_metrics_fn,
    )

    # Log detailed metrics
    logger.info("Evaluation Results:")
    for metric_name, value in test_metrics.items():
        logger.info(f"{metric_name}: {value}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TrOCR model.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path on HuggingFace Hub",
    )
    parser.add_argument(
        "--census_data", action="store_true", help="Use census data for evaluation"
    )
    parser.add_argument("--private_repo", type=str, help="Private dataset repository")
    parser.add_argument(
        "--max_label_length", type=int, default=64, help="Maximum label length"
    )

    args = parser.parse_args()

    # Run evaluation
    _ = evaluate_model(
        model_name=args.model_name,
        census_data=args.census_data,
        private_repo=args.private_repo,
        max_label_length=args.max_label_length,
    )


if __name__ == "__main__":
    main()
