import argparse

from dotenv import load_dotenv
from huggingface_hub import HfApi

from trocr_handwritten.parse.utils import _get_hf_token
from trocr_handwritten.utils.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


def push_model_to_hub(model_path, repo_id, filename, private=True):
    """
    Push a trained YOLO model to HuggingFace Hub.

    Args:
        model_path: Local path to the .pt model file
        repo_id: HuggingFace repository ID (e.g. "agomberto/historical-layout-ft")
        filename: Target filename on HuggingFace Hub
        private: Whether the repository should be private

    Returns:
        str: URL of the uploaded file
    """
    token = _get_hf_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    url = api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    logger.info(f"Model pushed to: {url}")
    return url


def main():
    """CLI entry point for pushing a trained model to HuggingFace Hub."""
    parser = argparse.ArgumentParser(
        description="Push trained model to HuggingFace Hub"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="agomberto/historical-layout-ft")
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--private", action="store_true", default=True)
    args = parser.parse_args()

    push_model_to_hub(args.model_path, args.repo_id, args.filename, args.private)


if __name__ == "__main__":
    main()
