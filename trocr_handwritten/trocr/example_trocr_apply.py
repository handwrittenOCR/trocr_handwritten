import logging
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List
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


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def plot_images_with_predictions(
    images: List[np.ndarray], texts: List[str], repo_name: str = "output"
):
    """
    Pour chaque image (non modifiée) :
      - Sauvegarde l'image d'origine en PNG.
      - Crée une page PDF dont la taille est:
           largeur = largeur de l'image
           hauteur = hauteur de l'image + marge pour le titre
      - Ajoute le titre (texte) dans la marge en haut (centré)
      - Ajoute l'image (non redimensionnée) juste en dessous
    Le PDF final contient autant de pages que d'images et aucune image n'est redimensionnée.
    """
    # Liste pour stocker les fichiers PNG temporaires
    fig, ax = plt.subplots(len(images), 1, figsize=(15, len(images) * 10))

    for idx, (text, image) in enumerate(zip(texts, images)):
        ax[idx].imshow(image, cmap="gray")
        ax[idx].set_title(text, fontsize=16)
        ax[idx].axis("off")

    plt.tight_layout(h_pad=0.1, w_pad=0.1)

    plt.savefig(f"predictions_{repo_name.split('/')[-1]}.png")
    plt.close()


def apply_model(
    model_name: str,
    num_samples: int = 20,
    seed: int = 42,
    hf_token: str = None,
    census_data: bool = True,
    private_repo: str = None,
    max_label_length: int = 64,
) -> None:
    """
    Apply the model to random samples and visualize results.

    Args:
        model_name: Name or path of the model on HuggingFace Hub
        num_samples: Number of random samples to process
        seed: Random seed for reproducibility
        hf_token: HuggingFace token for private repositories
        census_data: Whether to use census data
        private_repo: Private dataset repository
        max_label_length: Maximum label length
    """
    # Set random seed
    set_random_seed(seed)

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
    test_dataset = datasets["test"]

    # Select random samples
    total_samples = len(test_dataset)
    random_indices = random.sample(
        range(total_samples), min(num_samples, total_samples)
    )

    # Process selected samples
    selected_images = []
    predicted_texts = []

    for idx in random_indices:
        # Get image and process it
        sample = test_dataset[idx]
        image = sample["image"]
        pixel_values = sample["pixel_values"].clone().detach().unsqueeze(0)

        # Generate prediction
        outputs = ocr_model.model.generate(
            pixel_values.to(ocr_model.model.device),
            max_length=max_label_length,
            num_beams=4,
            early_stopping=True,
        )

        # Decode prediction
        predicted_text = ocr_model.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        selected_images.append(image)
        predicted_texts.append(predicted_text)

        logger.info(f"Processed sample {idx}: {predicted_text}")

    # Plot results
    plot_images_with_predictions(selected_images, predicted_texts, repo_name=model_name)


def main():
    parser = argparse.ArgumentParser(
        description="Apply and visualize TrOCR model predictions."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path on HuggingFace Hub",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of random samples to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--census_data",
        action="store_true",
        help="Use census data for evaluation",
    )
    parser.add_argument(
        "--private_repo",
        type=str,
        help="Private dataset repository",
    )
    parser.add_argument(
        "--max_label_length",
        type=int,
        default=64,
        help="Maximum label length",
    )

    args = parser.parse_args()

    # Run application
    apply_model(
        model_name=args.model_name,
        num_samples=args.num_samples,
        seed=args.seed,
        census_data=args.census_data,
        private_repo=args.private_repo,
        max_label_length=args.max_label_length,
    )


if __name__ == "__main__":
    main()
