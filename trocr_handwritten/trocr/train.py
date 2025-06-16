import logging
from transformers import TrainerCallback
from huggingface_hub import login

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


class PushToHubCallback(TrainerCallback):
    """Custom callback to push model, tokenizer, and processor to Hub after each epoch."""

    def __init__(self, ocr_model, hub_model_id, huggingface_api_key, private=True):
        self.ocr_model = ocr_model
        self.hub_model_id = hub_model_id
        self.huggingface_api_key = huggingface_api_key
        self.private = private

    def on_epoch_end(self, args, state, control, **kwargs):
        """Push model, tokenizer, and processor to Hub after each epoch."""
        epoch = state.epoch
        logger.info(f"Pushing model to Hub after epoch {epoch}")

        try:
            # Push model
            self.ocr_model.model.push_to_hub(
                repo_id=self.hub_model_id,
                token=self.huggingface_api_key,
                commit_message=f"Update model after epoch {epoch}",
                private=self.private,
            )

            # Push tokenizer
            self.ocr_model.tokenizer.push_to_hub(
                repo_id=self.hub_model_id,
                token=self.huggingface_api_key,
                commit_message=f"Update tokenizer after epoch {epoch}",
            )

            # Push processor
            self.ocr_model.processor.push_to_hub(
                repo_id=self.hub_model_id,
                token=self.huggingface_api_key,
                commit_message=f"Update processor after epoch {epoch}",
            )

            logger.info(
                f"Successfully pushed all components to Hub after epoch {epoch}"
            )
        except Exception as e:
            logger.error(f"Error pushing to Hub after epoch {epoch}: {str(e)}")

        return control


def train_model():
    """Main function to train the OCR model."""

    # Load settings
    train_settings = TrainSettings()
    ocr_model_settings = OCRModelSettings()
    trainer_datasets_settings = TrainerDatasetsSettings()

    # Login to Hugging Face Hub if API key is available
    if ocr_model_settings.huggingface_api_key:
        login(token=ocr_model_settings.huggingface_api_key)
        logger.info("Successfully logged in to Hugging Face Hub")

    # Create the OCRModel instance
    ocr_model = OCRModel(settings=ocr_model_settings, train_settings=train_settings)

    # Create the TrainerDatasets instance
    trainer_datasets = TrainerDatasets(
        settings=trainer_datasets_settings,
        tokenizer=ocr_model.tokenizer,
        processor=ocr_model.processor,
        preprocess_images=trainer_datasets_settings.preprocess_images,
    )

    # Load and process the data
    datasets = trainer_datasets.load_and_process_data()

    # Set up the compute_metrics function
    compute_metrics_fn = OCRModel.setup_compute_metrics(
        ocr_model.tokenizer, ocr_model.processor
    )

    # Create PushToHubCallback if pushing to hub is enabled
    callbacks = []
    if train_settings.push_to_hub and train_settings.hub_model_id:
        push_to_hub_callback = PushToHubCallback(
            ocr_model=ocr_model,
            hub_model_id=train_settings.hub_model_id,
            huggingface_api_key=ocr_model_settings.huggingface_api_key,
            private=train_settings.private_hub_repo,
        )
        callbacks.append(push_to_hub_callback)

    # Train the model
    result, trainer = ocr_model.train(
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics_fn=compute_metrics_fn,
        callbacks=callbacks,
    )

    # Evaluate the model on the test dataset
    test_metrics = ocr_model.evaluate(
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        test_dataset=datasets["test"],
        compute_metrics_fn=compute_metrics_fn,
    )

    logger.info(f"Test metrics: {test_metrics}")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    train_model()
