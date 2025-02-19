import os
import json
from PIL import Image
from prepare_training_OCR.postprocess_lines import crop_line_image  # adjust function name if needed

def process_rejected_images(decisions_path: str, output_dir: str):
    """
    Loads a JSON file containing image decisions, then for every entry under the
    "rejected" key, applies the cropping transformation (crop_line_image) to its source image.
    """
    with open(decisions_path, "r") as f:
        decisions = json.load(f)
    
    # Get the list of rejected decisions
    rejected = decisions.get("rejected", [])
    if not rejected:
        print("No rejected images found in decisions.")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for entry in rejected:
        source_image = entry.get("source_image")
        if not source_image:
            print("No source_image for one of the rejected entries; skipping.")
            continue

        if not os.path.exists(source_image):
            print(f"Source image not found: {source_image}")
            continue

        print(f"Processing image: {source_image}")
        try:
            image = Image.open(source_image)
        except Exception as e:
            print(f"Error opening image {source_image}: {e}")
            continue
        
        # Apply the transformation from postprocess_lines.py (e.g., cropping)
        try:
            processed_image = crop_line_image(image)
        except Exception as e:
            print(f"Error processing image {source_image}: {e}")
            continue

        # Save the processed image with a modified filename (e.g., appending '_processed')
        base_name, ext = os.path.splitext(os.path.basename(source_image))
        output_path = os.path.join(output_dir, f"{base_name}_processed{ext}")
        try:
            processed_image.save(output_path)
            print(f"Saved processed image to {output_path}")
        except Exception as e:
            print(f"Error saving processed image {output_path}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Apply line-cropping transformation to rejected images listed in image_decisions."
    )
    parser.add_argument(
        "--decisions",
        type=str,
        default="image_decisions.json",
        help="Path to the image decisions JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="processed",
        help="Directory to save processed images."
    )
    args = parser.parse_args()
    
    process_rejected_images(args.decisions, args.output)

if __name__ == "__main__":
    main() 