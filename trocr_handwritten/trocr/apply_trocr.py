# 24/10/2024: proposed new version of apply_trocr.py, which does the following:
# - processes a folder at a time
# - reports results by folder
# - processes one image at a time
# - allows to specify a batch size

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import logging
import argparse
import torch
from PIL import Image
from os.path import join
from os import listdir, makedirs
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_folder(
    folder_path, output_path, model, processor, tokenizer, device, batch_size
):
    images_files = [x for x in listdir(folder_path) if x.lower().endswith(".jpg")]
    images_files.sort()

    makedirs(output_path, exist_ok=True)
    output_file = join(output_path, "ocrized.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for i in tqdm(
            range(0, len(images_files), batch_size), desc=f"Processing {folder_path}"
        ):
            batch_files = images_files[i : i + batch_size]
            images = [
                Image.open(join(folder_path, img)).convert("RGB") for img in batch_files
            ]
            bboxes = [
                [float(y) for y in x[:-4].split("_line_")[1].split("_")]
                for x in batch_files
            ]

            pixel_values = processor(
                images=images, return_tensors="pt"
            ).pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for image, bbox, generated_text in zip(
                batch_files, bboxes, generated_texts
            ):
                bbox_str = "_".join(map(str, bbox))
                f.write(f"{generated_text}\t{image}\t{bbox_str}\n")
                logging.info(f"{image}: {generated_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a handwritten OCR model.")
    parser.add_argument("--PATH_DATA", type=str, help="Path to the data files")
    parser.add_argument("--PATH_OUTPUT", type=str, help="Path to the output directory")
    parser.add_argument(
        "--trocr_model",
        type=str,
        default="agomberto/trocr-base-handwritten-fr",
        help="Path to the model",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="microsoft/trocr-large-handwritten",
        help="Path to the processor",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing"
    )

    args = parser.parse_args()

    logging.info("Loading model and processor...")
    processor = TrOCRProcessor.from_pretrained(args.processor)
    model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model)
    tokenizer = AutoTokenizer.from_pretrained(args.trocr_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    folders = [x for x in listdir(args.PATH_DATA) if "." not in x]
    for folder in tqdm(folders, desc="Processing folders"):
        folder_path = join(args.PATH_DATA, folder)
        output_path = join(args.PATH_OUTPUT, folder)
        process_folder(
            folder_path,
            output_path,
            model,
            processor,
            tokenizer,
            device,
            args.batch_size,
        )

    logging.info("OCR process completed.")
