 """
This script loads the validated cropped dataset (produced by your previous script)
and pushes it to the Hugging Face Hub. Each push (with a commit message) will be versioned,
so you can later review the history and changes.
"""

import os
from datasets import load_from_disk

# Path to your validated cropped dataset on disk.
dataset_path = "prepare_training_OCR/validated_cropped_dataset"

print(f"Loading the dataset from {dataset_path}...")
dataset = load_from_disk(dataset_path)

# Optionally, log in to your Hugging Face account using your HF token.
# Ensure you have set HF_TOKEN as an environment variable.
from huggingface_hub import login
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None:
    print("Please set your HUGGINGFACE_TOKEN environment variable before pushing.")
    exit(1)
login(token=hf_token)

# Push the dataset to the Hub.
# Replace "your-username/validated-cropped-dataset" with your desired repository name.
repo_id = "MarieBgl/handwritten-ocr-dataset_cropped_corrected_v20250219"
commit_message = "Initial commit of validated cropped dataset"
print(f"Pushing dataset to the Hub repo {repo_id} with commit message: '{commit_message}'")
dataset.push_to_hub(repo_id, commit_message=commit_message)

print("Dataset successfully pushed to the Hugging Face Hub!")