from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
)
import logging
import argparse
import torch
from PIL import Image
from os.path import join
from os import listdir, makedirs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a handwritten OCR model.")
    parser.add_argument("--PATH_DATA", type=str, help="Path to the data files")
    parser.add_argument(
        "--trocr_model",
        type=str,
        help="Path to the model",
        default="agomberto/trocr-base-handwritten-fr",
    )
    parser.add_argument(
        "--processor",
        type=str,
        help="Path to the processor",
        default="microsoft/trocr-large-handwritten",
    )
    parser.add_argument(
        "--PATH_OUTPUT",
        type=str,
        help="Path to the output file",
    )

    args = parser.parse_args()

    logging.info("Loading model and processor...")
    processor = TrOCRProcessor.from_pretrained(args.processor)

    model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model)
    tokenizer = AutoTokenizer.from_pretrained(args.trocr_model)

    # device gpu if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # get all the folders in the data - we need this because of the "columns" subfolders
    folders = [x for x in listdir(args.PATH_DATA) if "." not in x]
    for folder in folders:
        makedirs(join(args.PATH_OUTPUT, folder), exist_ok=True)
        subfolders = [x for x in listdir(join(args.PATH_DATA, folder)) if "." not in x]
        for subfolder in subfolders:
            makedirs(join(args.PATH_OUTPUT, folder, subfolder), exist_ok=True)

            logging.info(f"Loading {folder}/{subfolder} images...")
            images_files = [
                x
                for x in listdir(join(args.PATH_DATA, folder, subfolder))
                if ".jpg" in x
            ]
            logging.info(f"{len(images_files)} images found.")
            images = [
                Image.open(join(args.PATH_DATA, folder, subfolder, x)).convert("RGB")
                for x in images_files
            ]

            logging.info("Generating texts...")
            pixel_values = (
                processor(images=images, return_tensors="pt").pixel_values
            ).to(device)
            generated_ids = model.generate(pixel_values)
            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            logging.info(
                "Writing output to {}".format(
                    join(args.PATH_OUTPUT, folder, subfolder, "ocrized.txt")
                )
            )
            logging.info("Generated texts:")
            with open(
                join(args.PATH_OUTPUT, folder, subfolder, "ocrized.txt"), "w"
            ) as f:
                for image, generated_text in zip(images_files, generated_texts):
                    logging.info(generated_text)
                    f.write(generated_text + "\t" + image + "\n")
