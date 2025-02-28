import logging
import argparse
import torch
from PIL import Image
import os
from typing import List, Dict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_lines_folders(root_dir: str) -> List[str]:
    """Find all folders named 'lines' recursively."""
    lines_folders = []
    for root, dirs, _ in os.walk(root_dir):
        if "lines" in dirs:
            lines_folders.append(os.path.join(root, "lines"))
    return lines_folders


def process_images_batch(
    image_paths: List[str],
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> Dict[str, str]:
    """Process a batch of images and return their transcriptions."""
    try:
        # Load and preprocess images
        images = [Image.open(path).convert("RGB") for path in image_paths]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(
            device
        )

        # Generate text
        with torch.no_grad():  # Disable gradient computation for inference
            generated_ids = model.generate(pixel_values)
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Create results dictionary
        results = {path: text for path, text in zip(image_paths, generated_texts)}
        return results
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return {path: "ERROR" for path in image_paths}


def process_folder(
    folder_path: str,
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> None:
    """Process all images in a folder and save results."""
    try:
        # Get all image files
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            logger.warning(f"No images found in {folder_path}")
            return

        # Process images in batches with progress bar
        results = {}
        for i in tqdm(
            range(0, len(image_files), batch_size),
            desc=f"Processing {os.path.basename(folder_path)}",
            unit="batch",
        ):
            batch = image_files[i : i + batch_size]
            batch_results = process_images_batch(
                batch, model, processor, tokenizer, device
            )
            results.update(batch_results)

        # Save results
        output_dir = os.path.dirname(folder_path)
        output_file = os.path.join(output_dir, "transcriptions.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            for image_path, text in results.items():
                f.write(f"{os.path.basename(image_path)}\t{text}\n")

        logger.info(f"Saved transcriptions to {output_file}")
    except Exception as e:
        logger.error(f"Error processing folder {folder_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Apply TrOCR model to images.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory to search for 'lines' folders",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/trocr-large-handwritten",
        help="Model name or path",
    )
    parser.add_argument(
        "--hf_token", type=str, help="HuggingFace token for private models"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing images"
    )
    args = parser.parse_args()

    # Login to HuggingFace if token provided
    if args.hf_token:
        login(token=args.hf_token)

    # Load model and processor
    logger.info(f"Loading model from {args.model_name}")
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set device and model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Find all 'lines' folders
    lines_folders = find_lines_folders(args.root_dir)
    logger.info(f"Found {len(lines_folders)} 'lines' folders to process")

    # Process folders sequentially
    for folder in tqdm(lines_folders, desc="Processing folders"):
        process_folder(folder, model, processor, tokenizer, device, args.batch_size)

    logger.info("All folders processed successfully!")


if __name__ == "__main__":
    main()
