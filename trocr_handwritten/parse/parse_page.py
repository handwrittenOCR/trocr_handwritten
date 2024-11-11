import argparse
from os.path import join, dirname, realpath
from os import listdir, makedirs
import json
from xml.dom import minidom
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import logging
from trocr_handwritten.utils.arunet_utils import (
    create_aru_net,
    get_test_loaders,
    load_checkpoint,
    save_test_predictions_as_imgs,
)

from trocr_handwritten.utils.parsing_lines import (
    get_coords,
    convert_coords,
    get_columns_coords,
    resize_columns,
    resize_bbox,
    bbox_split,
    is_overlap_picture,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

dir_path = dirname(realpath(__file__))

with open(join(dir_path, "config.json")) as f:
    config = json.load(f)

model_kwargs = dict(
    scale_space_num=config.get("SCALE_SPACE_NUM", 6),
    res_depth=config.get("RES_DEPTH", 3),
    feat_root=config.get("FEAT_ROOT", 8),
    filter_size=config.get("FILTER_SIZE", 3),
    pool_size=config.get("POOL_SIZE", 2),
    activation_name=config.get("ACTIVATION_NAME", "relu"),
    model=config.get("MODEL", "aru"),
    workers=config.get("NUM_WORKERS", 16),
    num_scales=config.get("NUM_SCALES", 5),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--PATH_PAGES", type=str, help="Path to the pages")
    parser.add_argument("--PATH_MODELS", type=str, help="Path to the models")
    parser.add_argument("--PATH_XML", type=str, help="Path to the XML files")
    parser.add_argument("--PATH_LINES", type=str, help="Path to save line images")
    parser.add_argument(
        "--verbose", type=bool, help="Display the image with the bboxes", default=True
    )

    args = parser.parse_args()

    logging.info("Creating ARU net...")
    model = create_aru_net(in_channels=1, out_channels=1, model_kwargs=model_kwargs).to(
        config.get("DEVICE", "cpu")
    )

    logging.info("Getting test loaders...")
    test_loader = get_test_loaders(
        args.PATH_PAGES,
        config.get("IMAGE_HEIGHT", 1024),
        config.get("IMAGE_WIDTH", 1024),
        config.get("PADDING", True),
        config.get("NUM_WORKERS"),
        config.get("PIN_MEMORY"),
    )

    logging.info("Loading checkpoint...")
    load_checkpoint(
        torch.load(
            join(args.PATH_MODELS, "cbad_2019.tar"),
            # FIXME: careful loading on device !!
            map_location=torch.device(config.get("DEVICE", "cpu")),
        ),
        model,
    )

    logging.info("Saving test predictions as XML...")
    save_test_predictions_as_imgs(
        test_loader,
        model,
        image_height=config.get("IMAGE_HEIGHT", 1024),
        image_width=config.get("IMAGE_WIDTH", 1024),
        padding=config.get("PADDING", True),
        output_dir=args.PATH_XML,
        # FIXME: careful loading on device !!    fmo
        device=config.get("DEVICE", "cpu"),
    )

    logging.info("---- XML files are created, now processing image by line...---")

    pages = {
        x.split(".")[0]: {} for x in listdir(args.PATH_PAGES) if "jpg" in x.lower()
    }

    ## procéder par batch
    for my_page in pages:
        logging.info(f"Processing page {my_page}...")
        data = minidom.parse(join(args.PATH_XML, f"{my_page}.xml"))
        coords, texts = get_coords(data, with_text=False)
        image = Image.open(join(args.PATH_PAGES, f"{my_page}.jpg")).convert("RGB")

        bboxes = []
        for coord in coords:
            bboxes.append(convert_coords(coord))

        coords, hist, indexes = get_columns_coords(bboxes, nbins=150, th=5)
        coords = resize_columns(coords, hist, indexes, th_width=200)
        _bboxes = [resize_bbox(bbox, coords) for bbox in bboxes]

        my_bboxes = []
        for bbox, index in _bboxes:
            my_bboxes += bbox_split(bbox, coords, index)

        columns = {}

        for bbox, i in my_bboxes[::-1]:
            if not columns.get(i):
                columns[i] = [bbox]
            else:
                if not is_overlap_picture(bbox, columns[i][-1], th=0.75):
                    columns[i].append(bbox)

        pages[my_page]["columns"] = columns

        for column, _bboxes in columns.items():
            # Create a directory for the column if it doesn't exist
            column_dir = join(args.PATH_LINES, my_page, f"column_{column}")
            print(column_dir)
            makedirs(column_dir, exist_ok=True)
            logging.info("Saving bbox images...")
            for i, bbox in enumerate(_bboxes):
                # Crop the image to the bbox
                bbox_image = image.crop(bbox)

                # Save the bbox image in the column directory
                bbox_image.save(join(column_dir, f"{my_page}_line_{i}.jpg"))

        if args.verbose:
            logging.info("Displaying image with bboxes...")
            image_with_bboxes = Image.open(
                join(args.PATH_PAGES, f"{my_page}.jpg")
            ).convert("RGB")
            draw = ImageDraw.Draw(image_with_bboxes, "RGB")

            colors = ["red", "blue", "yellow", "green", "orange"]

            for column, _bboxes in columns.items():
                for j, bbox in enumerate(_bboxes):
                    draw.rectangle(bbox, outline=colors[column], width=5)

            # display image_with_bboxes:
            plt.imshow(image_with_bboxes)
            plt.show()
