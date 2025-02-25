import logging
import argparse

from trocr_handwritten.trocr.dataset import TrainerDatasets
from trocr_handwritten.trocr.model import OCRModel
from trocr_handwritten.trocr.settings import (
    TrainSettings,
    OCRModelSettings,
    TrainerDatasetsSettings,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a handwritten OCR model.")
    parser.add_argument(
        "--config_file", type=str, help="Path to the config file", default="./config"
    )
    args = parser.parse_args()

    # Load settings from the config file
    train_settings = TrainSettings()
    ocr_model_settings = OCRModelSettings()
    trainer_datasets_settings = TrainerDatasetsSettings()

    # Create the OCRModel instance
    ocr_model = OCRModel(settings=ocr_model_settings, train_settings=train_settings)

    # Create the TrainerDatasets instance
    trainer_datasets = TrainerDatasets(
        settings=trainer_datasets_settings,
        tokenizer=ocr_model.tokenizer,
        processor=ocr_model.processor,
    )

    # Load and process the data
    datasets = trainer_datasets.load_and_process_data()

    # Set up the compute_metrics function
    compute_metrics_fn = OCRModel.setup_compute_metrics(
        ocr_model.tokenizer, ocr_model.processor
    )

    # Train the model
    result, trainer = ocr_model.train(
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics_fn=compute_metrics_fn,
    )

    # Evaluate the model on the test dataset
    test_metrics = ocr_model.evaluate(trainer, datasets["test"])

    logger.info(f"Test metrics: {test_metrics}")

    # Push to Hub if requested and not already done by trainer
    if train_settings.push_to_hub and train_settings.hub_model_id:
        logger.info(f"Pushing model to Hugging Face Hub: {train_settings.hub_model_id}")
        ocr_model.push_to_hub(
            trainer=trainer,
            repo_name=train_settings.hub_model_id,
            huggingface_api_key=ocr_model_settings.huggingface_api_key,
            private=train_settings.private_hub_repo,
        )

    logger.info("Training completed successfully!")
