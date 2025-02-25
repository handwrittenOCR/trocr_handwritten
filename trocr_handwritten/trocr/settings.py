from pydantic import BaseModel, Field
from typing import Optional
import os


class OCRModelSettings(BaseModel):
    model_name: str = Field(
        default="microsoft/trocr-large-handwritten",
        description="Name of the pre-trained model to use",
    )
    config_path: str = Field(
        default="./config", description="Path to the configuration files"
    )
    hub_repo: Optional[str] = Field(
        default=None, description="Hugging Face Hub repository to load the model from"
    )
    huggingface_api_key: Optional[str] = Field(
        default=None,
        description="Hugging Face API key for accessing private repositories",
    )


class TrainerDatasetsSettings(BaseModel):
    census_data: bool = Field(
        default=True, description="Whether to use the census dataset"
    )
    private_repo: Optional[str] = Field(
        default="agomberto/handwritten-ocr-dataset",
        description="Path to a private repository",
    )
    max_label_length: int = Field(
        default=64, description="Maximum length of the tokenized text labels"
    )
    huggingface_api_key: Optional[str] = Field(
        default=os.getenv("HUGGINGFACE_API_KEY"),
        description="Hugging Face API key for accessing private repositories",
    )


class TrainingConfig(BaseModel):
    predict_with_generate: bool = Field(
        default=True, description="Whether to predict with generate"
    )
    evaluation_strategy: str = Field(default="epoch", description="Evaluation strategy")
    save_strategy: str = Field(default="epoch", description="Save strategy")
    per_device_train_batch_size: int = Field(
        default=16, description="Batch size per device during training"
    )
    per_device_eval_batch_size: int = Field(
        default=16, description="Batch size per device during evaluation"
    )
    gradient_accumulation_steps: int = Field(
        default=8, description="Number of gradient accumulation steps"
    )
    optim: str = Field(default="adafactor", description="Optimizer to use")
    num_train_epochs: int = Field(default=20, description="Number of training epochs")
    fp16: bool = Field(
        default=False, description="Whether to use 16-bit floating point precision"
    )
    learning_rate: float = Field(default=4e-5, description="Learning rate")
    output_dir: str = Field(default="./", description="Output directory")
    logging_dir: str = Field(default="./logs", description="Logging directory")
    metric_for_best_model: str = Field(
        default="eval_loss", description="Metric for selecting the best model"
    )


class BeamConfig(BaseModel):
    max_length: int = Field(default=64, description="Maximum length for beam search")
    early_stopping: bool = Field(
        default=True, description="Whether to stop early in beam search"
    )
    no_repeat_ngram_size: int = Field(default=3, description="No repeat n-gram size")
    length_penalty: float = Field(
        default=2.0, description="Length penalty for beam search"
    )
    num_beams: int = Field(default=4, description="Number of beams for beam search")


class TrainSettings(BaseModel):
    output_dir: str = Field(
        default="./results", description="Directory to save the model"
    )
    push_to_hub: bool = Field(
        default=True, description="Whether to push the model to the Hugging Face Hub"
    )
    hub_model_id: Optional[str] = Field(
        default=None, description="The model ID on the Hugging Face Hub"
    )
    private_hub_repo: bool = Field(
        default=True, description="Whether the Hub repository should be private"
    )
    load_from_hub: Optional[str] = Field(
        default=None, description="Load a model from the Hugging Face Hub"
    )
    model_name: str = Field(
        default="microsoft/trocr-large-handwritten",
        description="Name of the pre-trained model to use if not loading from hub",
    )
    training_config: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration settings"
    )
    beam_config: BeamConfig = Field(
        default_factory=BeamConfig, description="Beam search configuration settings"
    )
