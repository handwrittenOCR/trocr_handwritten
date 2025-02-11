from pathlib import Path
from line_segmenter.image_processor import ImageProcessor
import urllib.request
from line_segmenter.settings import SegmentationSettings


def download_model(model_path: Path):
    """Download the Kraken blla model if it doesn't exist"""
    if not model_path.exists():
        print(f"Downloading model to {model_path}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/mittagessen/kraken/raw/main/kraken/blla.mlmodel"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully!")


def main(settings: SegmentationSettings):
    # Configure paths
    model_path = settings.model_path

    # Download model if needed
    download_model(model_path)

    input_dir = settings.input_dir

    # Create and run the processor
    image_folders = list(input_dir.glob("*"))
    for image_folder in image_folders:
        processor = ImageProcessor(str(model_path), settings)
        processor.process_directory(str(image_folder))


if __name__ == "__main__":
    settings = SegmentationSettings()
    main(settings)
