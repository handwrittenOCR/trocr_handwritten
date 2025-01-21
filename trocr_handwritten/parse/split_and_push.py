from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)

class_ids = {
    "Title": 0,
    "En-tête": 1,
    "Marge": 2,
    "Nom": 3,
    "Plein Texte": 4,
    "Signature": 5,
    "Table": 6,
    "Section": 7,
}


def convert_txt_to_jsonl(txt_path):
    """
    Convert a single txt file to JSONL format

    Args:
        txt_path (str): Path to the txt file

    Returns:
        dict: JSONL entry
    """
    # Get corresponding image name
    # find the image name in the path_to_images directory

    img_name = txt_path.stem + ".jpg"

    # Read the txt file
    bboxes = []
    categories = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            my_id = " ".join(line.split()[:-4])
            try:
                class_id = int(my_id)
            except ValueError:
                class_id = class_ids[my_id]
            x_center, y_center, width, height = map(float, line.strip().split()[-4:])

            # Convert to absolute coordinates
            bbox = [x_center, y_center, width, height]

            bboxes.append(bbox)
            categories.append(int(class_id))

    # Create JSONL entry with the original image name
    json_entry = {
        "file_name": img_name,  # Use the img_name with the correct case
        "objects": {"bbox": bboxes, "categories": categories},
    }

    return json_entry


def convert_directory(path_to_images, txt_dir, output_path):
    """
    Convert all txt files in directory to single JSONL file

    Args:
        path_to_images (str): Path to the images directory
        txt_dir (str): Path to the txt files directory
        output_path (str): Path to the output JSONL file
    """
    path_to_images = Path(path_to_images)  # Ensure path_to_images is a Path object
    txt_files = Path(txt_dir).glob("*.txt")
    with open(output_path, "w") as f:
        for txt_file in txt_files:
            json_entry = convert_txt_to_jsonl(txt_file)
            if json_entry:  # Only write if json_entry is not None
                f.write(json.dumps(json_entry) + "\n")


def main(path_to_images, test_size=0.2):
    """
    Main function to split the dataset and push to HuggingFace Hub

    Args:
        path_to_images (str): Path to the images directory
        test_size (float): Proportion of the dataset to include in the test split
    """
    # Load the dataset locally
    dataset = load_dataset("imagefolder", data_dir=path_to_images)
    # Convert to pandas for easier splitting
    df = dataset["train"].to_pandas()
    print(f"df: {df.head()}")

    # Perform stratified split
    train_idx, test_idx = train_test_split(
        range(len(df)),
        test_size=test_size,
        random_state=42,
    )

    # Create new dataset splits
    train_dataset = dataset["train"].select(train_idx)
    test_dataset = dataset["train"].select(test_idx)

    # Combine into a DatasetDict
    dataset_split = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Verify the split
    logging.info(f"Train set size: {len(train_dataset)}")
    logging.info(f"Test set size: {len(test_dataset)}")

    # Optional: verify distribution of labels in both splits
    categories = []
    for i in range(len(train_dataset["objects"])):
        categories += train_dataset["objects"][i]["categories"]
    categories_train = np.unique(categories, return_counts=True)[1] / len(categories)

    categories = []
    for i in range(len(test_dataset["objects"])):
        categories += test_dataset["objects"][i]["categories"]
    categories_test = np.unique(categories, return_counts=True)[1] / len(categories)

    logging.info("\nLabel distribution in train set: %s", categories_train)
    logging.info("Label distribution in test set: %s", categories_test)

    return dataset_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_images",
        type=str,
        default="data/20250111_bagnards_annotated/images/",
    )
    parser.add_argument("--repo_name", type=str, default="historical-layout-bagnards")
    parser.add_argument("--username", type=str, default="MarieBgl")
    parser.add_argument(
        "--path_to_labels",
        type=str,
        default="data/20250111_bagnards_annotated/labels/",
    )
    parser.add_argument(
        "--path_to_output",
        type=str,
        default="data/20250111_bagnards_annotated/images/metadata.jsonl",
    )
    args = parser.parse_args()

    convert_directory(args.path_to_images, args.path_to_labels, args.path_to_output)

    dataset_split = main(args.path_to_images)

    dataset_split.push_to_hub(f"{args.username}/{args.repo_name}", private=True)
