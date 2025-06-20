import logging
from typing import Dict, Any, Tuple, Callable, Optional, List
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from transformers import TrOCRProcessor, AutoTokenizer, VisionEncoderDecoderModel
from huggingface_hub import login
from transformers import GenerationConfig

import evaluate
from trocr_handwritten.trocr.settings import OCRModelSettings, TrainSettings
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrOCRDataCollator:
    """
    Data collator for TrOCR models. Handles pixel_values and labels correctly.
    """

    def __init__(self, processor=None, tokenizer=None):
        self.processor = processor
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Remove any unexpected fields from the features
        for feature in features:
            feature.pop("num_items_in_batch", None)

        # Prepare batch
        batch = {}

        # Stack pixel_values
        if "pixel_values" in features[0]:
            pixel_values = [feature["pixel_values"] for feature in features]
            batch["pixel_values"] = torch.stack(pixel_values)

        # Stack labels if present
        if "labels" in features[0]:
            labels = [feature["labels"] for feature in features]
            # Pad labels to max length
            max_label_length = max(len(label) for label in labels)
            padded_labels = []
            for label in labels:
                # Ensure labels are Long tensors
                if not isinstance(label, torch.LongTensor) and not isinstance(
                    label, torch.cuda.LongTensor
                ):
                    label = label.long()
                padding = [-100] * (max_label_length - len(label))
                padded_labels.append(
                    torch.cat([label, torch.tensor(padding, dtype=torch.long)])
                )
            batch["labels"] = torch.stack(padded_labels)

        # Ensure decoder_input_ids are properly created if needed
        if "decoder_input_ids" not in batch and "labels" in batch:
            batch["decoder_input_ids"] = self._shift_tokens_right(
                batch["labels"],
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
            )

        return batch

    def _shift_tokens_right(self, input_ids, pad_token_id, decoder_start_token_id):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        # Replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class TrOCRTrainer(Seq2SeqTrainer):
    """
    Custom trainer for TrOCR models that handles the num_items_in_batch issue.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Override compute_loss to remove num_items_in_batch from inputs.
        """
        # Remove num_items_in_batch from inputs
        if "num_items_in_batch" in inputs:
            inputs.pop("num_items_in_batch")

        # Call the parent class's compute_loss method
        return super().compute_loss(model, inputs, return_outputs)


class OCRModel:
    """
    A class to handle OCR model configuration, training, and evaluation.
    """

    def __init__(
        self,
        settings: OCRModelSettings,
        train_settings: TrainSettings,
        evaluate_only: bool = False,
    ):
        """
        Initialize the OCRModel class.

        Args:
            settings: OCRModelSettings instance containing model configuration.
            train_settings: TrainSettings instance containing training configuration.
        """
        self.settings = settings
        self.train_settings = train_settings
        self.evaluate_only = evaluate_only
        # Load from Hub if repository is provided
        if self.settings.hub_repo:
            logger.info(
                f"Loading model from Hugging Face Hub: {self.settings.hub_repo}"
            )
            try:
                self.load_from_hub(
                    self.settings.hub_repo,
                    self.settings.processor_name,
                    self.settings.huggingface_api_key,
                )
            except Exception as e:
                logger.error(f"Error loading model from Hub: {str(e)}")
                logger.warning("Falling back to default model...")
                self._load_default_model()
        # Use provided model, tokenizer, and processor if available
        else:
            logger.info("Loading default model...")
            self._load_default_model()

        self._set_model_params()
        self.training_args = self._set_training_args()

    def _load_default_model(self):
        """
        Load the default model, tokenizer, and processor.
        """
        logger.info(f"Loading default model: {self.settings.model_name}")
        try:
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.settings.model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
            self.processor = TrOCRProcessor.from_pretrained(
                self.settings.processor_name,
                use_fast=True,
            )
            logger.info(
                f"Successfully loaded default model: {self.settings.model_name}"
            )
        except Exception as e:
            logger.error(f"Error loading default model: {str(e)}")
            raise ValueError("Failed to load default model and no model was provided.")

    def train(
        self,
        train_dataset,
        eval_dataset,
        compute_metrics_fn: Optional[Callable] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> Tuple[Any, TrOCRTrainer]:
        """
        Train the model.

        Args:
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            compute_metrics_fn: A function to compute metrics during evaluation.

        Returns:
            Tuple[Any, TrOCRTrainer]: The training results and the trainer object.
        """
        trainer = TrOCRTrainer(
            model=self.model,
            processing_class=self.processor,
            args=self.training_args,
            compute_metrics=compute_metrics_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            data_collator=TrOCRDataCollator(
                processor=self.processor, tokenizer=self.tokenizer
            ),
        )

        logger.info("Starting training...")
        result = trainer.train()

        return result, trainer

    def evaluate(
        self,
        train_dataset,
        eval_dataset,
        test_dataset,
        compute_metrics_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on the test dataset.

        Args:
            trainer: The trainer to use for evaluation.
            test_dataset: The test dataset.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        trainer = TrOCRTrainer(
            model=self.model,
            processing_class=self.processor,
            args=self.training_args,
            compute_metrics=compute_metrics_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=TrOCRDataCollator(
                processor=self.processor, tokenizer=self.tokenizer
            ),
        )
        logger.info("Evaluating on test dataset...")
        metrics = trainer.evaluate(test_dataset)

        return metrics

    def _create_model_card(self, metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Create a model card with performance metrics.

        Args:
            metrics: Evaluation metrics to include in the model card.

        Returns:
            str: The model card content.
        """
        self.settings.model_name.split("/")[-1]

        # Basic model card template
        model_card = f"""---
                    language: en
                    license: apache-2.0
                    tags:
                    - trocr
                    - ocr
                    - handwritten-text-recognition
                    datasets:
                    - agomberto/FrenchCensus-handwritten-texts
                    ---

                    # TrOCR for Handwritten Text Recognition

                    This model is a fine-tuned version of [{self.settings.model_name}](https://huggingface.co/{self.settings.model_name}) for handwritten text recognition.

                    ## Model description

                    This model uses the TrOCR architecture, which combines a vision encoder with a text decoder to recognize text in images.

                    ## Performance

                    """

        # Add metrics if available
        if metrics:
            model_card += "### Evaluation Results\n\n"
            model_card += "| Metric | Value |\n"
            model_card += "|--------|-------|\n"

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    model_card += f"| {metric_name} | {metric_value:.4f} |\n"
                else:
                    model_card += f"| {metric_name} | {metric_value} |\n"

        model_card += """
                    ## Usage

                    ```python
                    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                    from PIL import Image

                    processor = TrOCRProcessor.from_pretrained("REPO_ID")
                    model = VisionEncoderDecoderModel.from_pretrained("REPO_ID")

                    # Load image
                    image = Image.open("path/to/image").convert("RGB")

                    # Process image
                    pixel_values = processor(image, return_tensors="pt").pixel_values

                    # Generate text
                    generated_ids = model.generate(pixel_values)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    print(generated_text)
                    ```
                    """

        return model_card

    def push_to_hub(
        self,
        repo_name: str,
        huggingface_api_key: str,
        private: bool = False,
        commit_message: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Push the trained model to the Hugging Face Hub.

        Args:
            repo_name: The name of the repository to push to.
            huggingface_api_key: The Hugging Face API key.
            private: Whether the repository should be private.
            commit_message: Optional commit message.
            metrics: Optional evaluation metrics to include in the model card.

        Returns:
            str: The URL of the pushed model on the Hub.
        """
        try:
            login(token=huggingface_api_key)
            logger.info("Successfully logged in to Hugging Face Hub")

            # Push the model to the Hub
            logger.info(f"Pushing model to {repo_name}...")

            # Set default commit message if not provided
            if commit_message is None:
                commit_message = (
                    "Upload model trained with TrOCR for handwritten text recognition"
                )

            # Create a model card with metrics
            model_card_content = self._create_model_card(metrics)

            # Create README.md file with model card content
            import os

            os.makedirs("./tmp_model_card", exist_ok=True)
            with open("./tmp_model_card/README.md", "w") as f:
                f.write(model_card_content)

            # Push the model with the model card
            self.model.push_to_hub(
                repo_id=repo_name,
                use_auth_token=huggingface_api_key,
                commit_message=commit_message,
                create_pr=False,
                private=private,
                card_data=model_card_content,
            )
            logger.info(f"Model successfully pushed to {repo_name}")

            # Push the tokenizer
            self.tokenizer.push_to_hub(
                repo_id=repo_name,
                use_auth_token=huggingface_api_key,
                commit_message="Upload TrOCR tokenizer",
                create_pr=False,
            )
            logger.info(f"Tokenizer successfully pushed to {repo_name}")

            # Push the processor
            self.processor.push_to_hub(
                repo_id=repo_name,
                use_auth_token=huggingface_api_key,
                commit_message="Upload TrOCR processor",
                create_pr=False,
            )
            logger.info(f"Processor successfully pushed to {repo_name}")

            return f"https://huggingface.co/{repo_name}"

        except ImportError:
            logger.error("huggingface_hub not installed. Cannot push to Hub.")
            raise
        except Exception as e:
            logger.error(f"Error pushing model to Hub: {str(e)}")
            raise

    def load_from_hub(
        self,
        repo_name: str,
        processor_name: str,
        huggingface_api_key: Optional[str] = None,
    ) -> "OCRModel":
        """
        Load a model from the Hugging Face Hub.

        Args:
            repo_name: The name of the repository to load from.
            processor_name: The name of the processor to load from.
            huggingface_api_key: Optional Hugging Face API key for private repositories.

        Returns:
            OCRModel: An OCRModel instance with the loaded model.
        """
        try:
            # Login to Hugging Face Hub if API key is provided
            if huggingface_api_key:
                try:
                    login(token=huggingface_api_key)
                    logger.info("Successfully logged in to Hugging Face Hub")
                except ImportError:
                    logger.warning(
                        "huggingface_hub not installed. Cannot login with API key."
                    )
                except Exception as e:
                    logger.warning(f"Failed to login to Hugging Face Hub: {str(e)}")

            # Load the model, tokenizer, and processor from the Hub
            logger.info(f"Loading model from {repo_name}...")
            self.model = VisionEncoderDecoderModel.from_pretrained(repo_name)

            logger.info(f"Loading tokenizer from {repo_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(repo_name)

            logger.info(f"Loading processor from {processor_name}...")
            self.processor = TrOCRProcessor.from_pretrained(
                processor_name, use_fast=True
            )

        except Exception as e:
            logger.error(f"Error loading model from Hub: {str(e)}")
            raise

    @staticmethod
    def setup_compute_metrics(tokenizer, processor) -> Callable:
        """
        Set up the compute_metrics function.

        Args:
            tokenizer: The tokenizer to use.
            processor: The processor to use.

        Returns:
            Callable: A function that computes metrics during evaluation.
        """
        try:

            cer_metric = evaluate.load("cer")
            wer_metric = evaluate.load("wer")

            return OCRModel.compute_metrics(
                cer_metric, wer_metric, tokenizer, processor
            )
        except ImportError:
            logger.warning("evaluate package not found. Metrics will not be computed.")
            return None

    def _set_training_args(self) -> Seq2SeqTrainingArguments:
        """
        Configure the training arguments from the settings.

        Returns:
            Seq2SeqTrainingArguments: The configured training arguments.
        """
        training_config = self.train_settings.training_config

        return Seq2SeqTrainingArguments(
            predict_with_generate=training_config.predict_with_generate,
            eval_strategy=training_config.eval_strategy,
            save_strategy=training_config.save_strategy,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            optim=training_config.optim,
            num_train_epochs=training_config.num_train_epochs,
            fp16=training_config.fp16,
            learning_rate=training_config.learning_rate,
            output_dir=training_config.output_dir,
            logging_dir=training_config.logging_dir,
            metric_for_best_model=training_config.metric_for_best_model,
            push_to_hub=False,
            hub_model_id=self.train_settings.hub_model_id,
            hub_token=self.settings.huggingface_api_key,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            evaluation_strategy="no" if self.evaluate_only else "epoch",
        )

    def _set_model_params(self) -> None:
        """
        Configure the model parameters from the settings.
        """
        beam_config = self.train_settings.beam_config

        # Set the special tokens used for creating the decoder input IDs from the labels
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        # Ensure the vocabulary size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # Set the beam search parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id

        generation_config = GenerationConfig(
            max_length=beam_config.max_length,
            early_stopping=beam_config.early_stopping,
            no_repeat_ngram_size=beam_config.no_repeat_ngram_size,
            length_penalty=beam_config.length_penalty,
            num_beams=beam_config.num_beams,
            pad_token_id=self.model.config.pad_token_id,
            bos_token_id=self.model.config.decoder_start_token_id,
            eos_token_id=self.model.config.eos_token_id,
        )

        self.model.generation_config = generation_config
        self.model.encoder.gradient_checkpointing_enable()
        self.model.decoder.gradient_checkpointing_enable()

    @staticmethod
    def compute_metrics(
        cer_metric, wer_metric, text_tokenizer, text_processor
    ) -> Callable:
        """
        Configure the metrics to use for evaluation.

        Args:
            cer_metric: The character error rate metric.
            wer_metric: The word error rate metric.
            text_tokenizer: The tokenizer to use for text tokenization.
            text_processor: The processor to use for text processing.

        Returns:
            Callable: A function that computes the metrics.
        """

        def compute_predictions(predictions):
            """
            Compute the metrics for the given predictions.

            Args:
                predictions: The predictions to compute the metrics for.

            Returns:
                Dict[str, float]: A dictionary containing the computed metrics.
            """
            try:
                # Get the label IDs and prediction IDs
                label_ids = predictions.label_ids
                prediction_ids = predictions.predictions

                # Decode the prediction IDs and label IDs
                prediction_strings = text_tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )

                # Replace -100 with pad token ID for proper decoding
                label_ids_copy = label_ids.copy()
                label_ids_copy[label_ids_copy == -100] = (
                    text_processor.tokenizer.pad_token_id
                )
                label_strings = text_tokenizer.batch_decode(
                    label_ids_copy, skip_special_tokens=True
                )

                # Compute the character error rate and word error rate
                character_error_rate = cer_metric.compute(
                    predictions=prediction_strings, references=label_strings
                )
                word_error_rate = wer_metric.compute(
                    predictions=prediction_strings, references=label_strings
                )

                return {"cer": character_error_rate, "wer": word_error_rate}
            except Exception as e:
                logger.error(f"Error computing metrics: {str(e)}")
                return {"cer": float("nan"), "wer": float("nan")}

        return compute_predictions

    def predict(
        self,
        train_dataset,
        eval_dataset,
        test_dataset,
        compute_metrics_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions on a dataset and calculate metrics for each item individually.

        Args:
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            test_dataset: The dataset to generate predictions on.
            compute_metrics_fn: Function to compute metrics (from setup_compute_metrics).

        Returns:
            Dict[str, Any]: A dictionary containing predictions, labels, and metrics.
        """
        logger.info(
            f"Generating predictions on dataset with {len(test_dataset)} items..."
        )

        # Create a trainer instance
        trainer = TrOCRTrainer(
            model=self.model,
            processing_class=self.processor,
            args=self.training_args,
            compute_metrics=compute_metrics_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=TrOCRDataCollator(
                processor=self.processor, tokenizer=self.tokenizer
            ),
        )

        # Get predictions from the trainer
        predictions = trainer.predict(test_dataset, metric_key_prefix="predict")

        # Get raw predictions and labels
        prediction_ids = predictions.predictions
        label_ids = predictions.label_ids

        # Calculate loss per item (if available)
        per_item_loss = None
        if hasattr(predictions, "losses") and predictions.losses is not None:
            per_item_loss = predictions.losses.tolist()

        # Decode predictions and labels
        predicted_texts = self.tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )

        # Replace -100 values in label_ids with pad_token_id for decoding
        label_ids_copy = label_ids.copy()
        label_ids_copy[label_ids_copy == -100] = self.processor.tokenizer.pad_token_id
        true_texts = self.tokenizer.batch_decode(
            label_ids_copy, skip_special_tokens=True
        )

        # Initialize results structure
        results = {
            "predictions": predicted_texts,
            "labels": true_texts,
            "item_metrics": [],
            "losses": per_item_loss,
        }

        # Calculate metrics for each item individually
        if compute_metrics_fn is not None:
            # Get metrics calculators from the existing setup
            cer_metric = evaluate.load("cer")
            wer_metric = evaluate.load("wer")

            # Process each prediction/label pair
            for idx, (pred_text, true_text) in enumerate(
                zip(predicted_texts, true_texts)
            ):
                # Calculate metrics for this individual item
                cer = cer_metric.compute(
                    predictions=[pred_text], references=[true_text]
                )
                wer = wer_metric.compute(
                    predictions=[pred_text], references=[true_text]
                )

                # Create basic metrics
                item_metrics = {
                    "prediction": pred_text,
                    "label": true_text,
                    "cer": cer,
                    "wer": wer,
                    "data_source": (
                        test_dataset[idx]["dataset_source"]
                        if "dataset_source" in test_dataset[idx]
                        else None
                    ),
                    "file_name": (
                        test_dataset[idx]["image_path"]
                        if "image_path" in test_dataset[idx]
                        else None
                    ),
                }

                # Add loss if available
                if per_item_loss is not None and idx < len(per_item_loss):
                    item_metrics["loss"] = per_item_loss[idx]

                results["item_metrics"].append(item_metrics)

        # Include overall metrics
        results["overall_metrics"] = {
            key.replace("predict_", ""): value
            for key, value in predictions.metrics.items()
        }

        return results
