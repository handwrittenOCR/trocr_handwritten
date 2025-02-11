from pathlib import Path
from line_segmenter.image_processor import ImageProcessor
import urllib.request


def download_model(model_path: Path):
    """Download the Kraken blla model if it doesn't exist"""
    if not model_path.exists():
        print(f"Downloading model to {model_path}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/mittagessen/kraken/raw/main/kraken/blla.mlmodel"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully!")


def main():
    # Configure paths
    current_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    model_path = current_dir / "models" / "blla.mlmodel"

    # Download model if needed
    download_model(model_path)

    input_dir = (
        current_dir / "data" / "processed" / "images" / "FRANOM58_078MIOM0870_0259"
    )

    # Create and run the processor
    processor = ImageProcessor(str(model_path))
    processor.process_directory(str(input_dir))


if __name__ == "__main__":
    main()
