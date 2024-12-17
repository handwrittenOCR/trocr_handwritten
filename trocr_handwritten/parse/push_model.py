from huggingface_hub import HfApi
from datetime import datetime


def push_to_hub(
    model_path,
    repo_id,
    commit_message=None,
):
    """Push trained model to Hugging Face Hub"""

    # Initialize Hugging Face API
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repository creation failed: {e}")
        return

    # Default commit message
    if commit_message is None:
        commit_message = (
            f"Upload model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # Upload the model file
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="model.pt",
            repo_id=repo_id,
            commit_message=commit_message,
        )
        print(f"Model successfully uploaded to {repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (username/repo-name)",
    )
    parser.add_argument("--commit-message", type=str, help="Commit message")

    args = parser.parse_args()

    push_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )
