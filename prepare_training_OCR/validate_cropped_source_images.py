"""
This script:
- Loads the HF dataset and the review decisions (from image_decisions.json).
- Processes only those source images that were "rejected" in the review.
- For each rejected source image, it stacks the original line images
  (as in review_source_images.py) on the left, and on the right stacks the cropped
  images (using crop_line_image from postprocess_lines.py). It then prompts the user
  to validate the cropping.
- If the cropping is validated, it updates each sample (by adding the "cropped_image"
  and corrected bounding box coordinates) and later combines those with the accepted ones.
- Finally, it writes out a new dataset (using datasets.Dataset.save_to_disk) that
  contains, for each sample, both the original metadata and the corrected (cropped) image data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from collections import defaultdict

from datasets import load_dataset, Dataset
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# Import the helper functions from your existing modules:

from prepare_training_OCR.review_source_images import reconstruct_source_image
from prepare_training_OCR.postprocess_lines import crop_line_image

def reconstruct_cropped_source_image(examples, crop_factor=0.1):
    """
    Given a list of samples (lines) belonging to a single source image,
    apply crop_line_image (to obtain cropped images with updated coordinates)
    and stack the cropped images vertically.
    """
    # Sort samples top-to-bottom using their original y1 coordinate.
    sorted_examples = sorted(examples, key=lambda ex: ex['y1'])
    
    # Process all samples (apply crop to each) and record sizes.
    cropped_images = []
    max_width = 0
    total_height = 0
    spacing = 20  # spacing between lines

    for ex in sorted_examples:
        cropped_img, _ = crop_line_image(ex, crop_factor=crop_factor)
        cropped_images.append(cropped_img)
        max_width = max(max_width, cropped_img.width)
        total_height += cropped_img.height
        total_height += spacing
    if cropped_images:
        total_height -= spacing  # remove extra spacing after the last line

    # Create the combined image
    combined_img = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for img in cropped_images:
        combined_img.paste(img, (0, y_offset))
        y_offset += img.height + spacing

    return combined_img

def apply_cropping_to_sample(sample, crop_factor=0.1):
    """
    Apply crop_line_image to a sample and update its dictionary
    by adding keys for the cropped image and updated coordinates.
    """
    cropped_img, new_coords = crop_line_image(sample, crop_factor=crop_factor)
    sample_updated = sample.copy()
    sample_updated["cropped_image"] = cropped_img
    sample_updated["cropped_x1"] = new_coords[0]
    sample_updated["cropped_y1"] = new_coords[1]
    sample_updated["cropped_x2"] = new_coords[2]
    sample_updated["cropped_y2"] = new_coords[3]
    return sample_updated

def main():
    output_path = "prepare_training_OCR/validated_cropped_dataset"
    # Parse command line arguments for adjustable crop factor.
    parser = argparse.ArgumentParser(description="Validate cropped source images with adjustable cropping.")
    parser.add_argument("--crop_factor", type=float, default=0.5, help="Initial crop factor value.")
    args = parser.parse_args()
    default_crop_factor = args.crop_factor

    # Check if an output dataset already exists to resume selection.
    existing_samples = []
    processed_source_ids = set()
    if os.path.exists(output_path):
        print("Found existing validated dataset. Resuming...")
        from datasets import load_from_disk
        existing_dataset = load_from_disk(output_path)
        existing_samples = list(existing_dataset)
        processed_source_ids = set(sample["source_image"] for sample in existing_samples)
    else:
        print("No existing validated dataset. Starting fresh.")

    # Load the original HF dataset (here we use the "train" split)
    print("Loading dataset...")
    dataset_dict = load_dataset("agomberto/handwritten-ocr-dataset")
    dataset = dataset_dict['train']

    # Load the review decisions; adjust the path if needed.
    decisions_file = "prepare_training_OCR/image_decisions.json"
    try:
        with open(decisions_file, "r") as f:
            decisions = json.load(f)
    except FileNotFoundError:
        print("Decisions file not found. Exiting ...")
        return

    # Identify rejected and accepted source images.
    rejected_source_ids = [src for src, dec in decisions.items() if dec["choice"] == "rejected"]
    accepted_source_ids = [src for src, dec in decisions.items() if dec["choice"] == "accepted"]
    print(f"Found {len(rejected_source_ids)} rejected and {len(accepted_source_ids)} accepted source images.")

    # Group samples by source_image, but skip images already processed.
    groups_rejected = defaultdict(list)
    groups_accepted = defaultdict(list)
    for sample in dataset:
        src_id = sample["source_image"]
        if src_id in processed_source_ids:
            continue
        if src_id in rejected_source_ids:
            groups_rejected[src_id].append(sample)
        elif src_id in accepted_source_ids:
            groups_accepted[src_id].append(sample)

    # List to store samples that have their cropping validated.
    cropping_validated_samples = []

    # Enable interactive plotting and create a persistent figure.
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Process rejected source images one by one.
    exit_flag = False
    for src_id, samples in groups_rejected.items():
        print(f"\nProcessing rejected source image: {src_id} with {len(samples)} lines")

        # Build the original stacked image view.
        original_combined = reconstruct_source_image(samples)

        # Set initial crop factor for this source image.
        current_crop_factor = default_crop_factor

        while True:
            # Build the corrected (cropped) stacked image view using current crop factor.
            corrected_combined = reconstruct_cropped_source_image(samples, crop_factor=current_crop_factor)

            # Update the persistent figure instead of creating a new one.
            axs[0].clear()
            axs[0].imshow(original_combined)
            axs[0].set_title(f"Original Combined Image: {src_id}")
            axs[0].axis("off")

            axs[1].clear()
            axs[1].imshow(corrected_combined)
            axs[1].set_title(f"Corrected (Cropped) Combined Image (crop_factor = {current_crop_factor}): {src_id}")
            axs[1].axis("off")

            fig.tight_layout()
            fig.canvas.draw_idle()
            plt.pause(0.001)

            # Ask for user action.
            user_input = input("Options: (V)alidate, (C)hange crop factor, (R)eject, (E)xit: ").strip().lower()

            if user_input == "v":
                print(f"Validated cropping for source image: {src_id} with crop_factor = {current_crop_factor}")
                # Update each sample from this source; apply the cropping transformation.
                for sample in samples:
                    updated_sample = apply_cropping_to_sample(sample, crop_factor=current_crop_factor)
                    cropping_validated_samples.append(updated_sample)
                break  # Accept current crop factor and move to next source image.
            elif user_input == "c":
                new_crop = input("Enter new crop factor (float value): ").strip()
                try:
                    new_crop_value = float(new_crop)
                    current_crop_factor = new_crop_value
                except ValueError:
                    print("Invalid crop factor value. Try again.")
                # Loop again with the updated crop factor.
            elif user_input == "r":
                print(f"Cropping rejected for source image: {src_id}. Skipping modifications...")
                break  # Do not update any samples for this source image.
            elif user_input == "e":
                print("Exiting early as per user request.")
                exit_flag = True
                break  # Exit the inner loop.
            else:
                print("Invalid input. Please try again.")

        if exit_flag:
            break

    # Combine the existing validated samples with the newly cropping-validated samples.
    new_dataset_samples = existing_samples + cropping_validated_samples
    total_validated = len(new_dataset_samples)
    print(f"\nTotal samples in new dataset (cropping validated only): {total_validated}")
    if total_validated == 0:
        print("No validated samples to save. Exiting without saving dataset.")
    else:
        # Define the set of keys we want to include in our final dataset
        desired_keys = [
            "source_image", "text", "x1", "y1", "x2", "y2",
            "image", "cropped_image", "cropped_x1", "cropped_y1", "cropped_x2", "cropped_y2"
        ]

        # Convert the list-of-dictionaries into a dictionary of columns using only desired keys.
        dataset_dict = { key: [sample.get(key) for sample in new_dataset_samples] for key in desired_keys }

        # Define features explicitly. The Image() feature helps process PIL.Image values.
        from datasets import Features, Value, Image
        features = Features({
            'source_image': Value('string'),
            'text': Value('string'),
            'x1': Value('float'),
            'y1': Value('float'),
            'x2': Value('float'),
            'y2': Value('float'),
            'image': Image(),            # original image; assumes it exists in every sample
            'cropped_image': Image(),    # cropped image produced by your function
            'cropped_x1': Value('float'),
            'cropped_y1': Value('float'),
            'cropped_x2': Value('float'),
            'cropped_y2': Value('float'),
        })

        new_dataset = Dataset.from_dict(dataset_dict, features=features)
        new_dataset.save_to_disk(output_path)
        print(f"New dataset saved to {output_path}")

if __name__ == "__main__":
    main() 