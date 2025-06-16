import logging
import argparse
import evaluate
import os
import torch
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
    rimes_data: bool = True,
    belfort_data: bool = True,
    private_repo: str = None,
    max_label_length: int = 64,
) -> dict:
    """
    Evaluate a model on test datasets.

    Args:
        model_name: Name or path of the model on HuggingFace Hub
        hf_token: HuggingFace token for private repositories
        census_data: Whether to use census data
        rimes_data: Whether to use Rimes data
        belfort_data: Whether to use Belfort data
        private_repo: Private dataset repository
        max_label_length: Maximum label length

    Returns:
        dict: Evaluation metrics for overall and per-dataset performance
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
        rimes_data=rimes_data,
        belfort_data=belfort_data,
        private_repo=private_repo,
        max_label_length=max_label_length,
        huggingface_api_key=hf_token,
    )

    # Initialize model
    logger.info(f"Loading model from {model_name}")
    ocr_model = OCRModel(
        settings=model_settings, train_settings=train_settings, evaluate_only=True
    )

    # Initialize datasets
    logger.info("Loading datasets")
    trainer_datasets = TrainerDatasets(
        settings=dataset_settings,
        tokenizer=ocr_model.tokenizer,
        processor=ocr_model.processor,
        evaluate_only=True,
    )

    # Load and process data
    datasets = trainer_datasets.load_and_process_data()

    # Set up metrics computation
    compute_metrics_fn = OCRModel.setup_compute_metrics(
        ocr_model.tokenizer, ocr_model.processor
    )

    # Initialize results dictionary
    all_results = {
        "per_dataset": {},
        "overall": {"predictions": [], "labels": [], "metrics": {}},
    }

    # Get dataset sources and indices
    dataset_indices = {}
    if hasattr(datasets["test"], "datasets"):
        current_idx = 0
        for dataset in datasets["test"].datasets:
            if hasattr(dataset, "dataset_source"):
                source = dataset.dataset_source
                if source not in dataset_indices:
                    dataset_indices[source] = []
                # Add indices for this dataset
                dataset_indices[source].extend(
                    range(current_idx, current_idx + len(dataset))
                )
                current_idx += len(dataset)

    # Track dataset sizes and losses for weighted average
    dataset_sizes = {}
    dataset_losses = {}

    # Evaluate each dataset individually
    for dataset_name in ["census", "rimes", "belfort", "private"]:
        if dataset_name in dataset_indices and dataset_indices[dataset_name]:
            logger.info(f"\nEvaluating {dataset_name.upper()} dataset...")

            # Create subset for this dataset
            dataset_subset = torch.utils.data.Subset(
                datasets["test"], dataset_indices[dataset_name]
            )

            # Get predictions and metrics for this dataset
            dataset_results = ocr_model.predict(
                train_dataset=None,
                eval_dataset=None,
                test_dataset=dataset_subset,
                compute_metrics_fn=compute_metrics_fn,
            )

            # Store dataset-specific results
            all_results["per_dataset"][dataset_name] = {
                "predictions": dataset_results["predictions"],
                "labels": dataset_results["labels"],
                "metrics": dataset_results["overall_metrics"],
            }

            # Track dataset size and loss for weighted average
            dataset_sizes[dataset_name] = len(dataset_results["predictions"])
            if "loss" in dataset_results["overall_metrics"]:
                dataset_losses[dataset_name] = dataset_results["overall_metrics"][
                    "loss"
                ]

            # Add to overall results
            all_results["overall"]["predictions"].extend(dataset_results["predictions"])
            all_results["overall"]["labels"].extend(dataset_results["labels"])

    # Compute overall metrics
    if all_results["overall"]["predictions"]:
        cer_metric = evaluate.load("cer")
        wer_metric = evaluate.load("wer")

        # Compute overall CER and WER
        overall_cer = cer_metric.compute(
            predictions=all_results["overall"]["predictions"],
            references=all_results["overall"]["labels"],
        )
        overall_wer = wer_metric.compute(
            predictions=all_results["overall"]["predictions"],
            references=all_results["overall"]["labels"],
        )

        # Compute weighted average loss
        total_size = sum(dataset_sizes.values())
        overall_loss = None
        if dataset_losses:
            overall_loss = sum(
                (dataset_losses[name] * size) / total_size
                for name, size in dataset_sizes.items()
                if name in dataset_losses
            )

        all_results["overall"]["metrics"] = {
            "cer": overall_cer,
            "wer": overall_wer,
            "loss": overall_loss,
        }

    # Log detailed metrics
    logger.info("\nOverall Evaluation Results:")
    for metric_name, value in all_results["overall"]["metrics"].items():
        logger.info(f"{metric_name}: {value}")

    logger.info("\nPer-Dataset Evaluation Results:")
    for dataset_name, results in all_results["per_dataset"].items():
        logger.info(f"\n{dataset_name.upper()} Dataset:")
        for metric_name, value in results["metrics"].items():
            logger.info(f"{metric_name}: {value}")

    return all_results


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
    parser.add_argument(
        "--rimes_data", action="store_true", help="Use Rimes data for evaluation"
    )
    parser.add_argument(
        "--belfort_data", action="store_true", help="Use Belfort data for evaluation"
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
        rimes_data=args.rimes_data,
        belfort_data=args.belfort_data,
        private_repo=args.private_repo,
        max_label_length=args.max_label_length,
    )


if __name__ == "__main__":
    main()
