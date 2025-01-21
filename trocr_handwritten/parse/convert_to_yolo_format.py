import os
from os.path import join
from PIL import Image
import pandas as pd
import argparse
import yaml

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


def process_image_set(df, path_folder, config):

    source_images_dir = join(path_folder, "images")
    print(f"source_images_dir: {source_images_dir}")
    output_dir = join(path_folder, "labels")
    unique_images = df["filename"].unique()
    class_ids = []

    for img_name in unique_images:
        img_annotations = df[df["filename"] == img_name]
        img = Image.open(join(source_images_dir, img_name))
        img_width = img.width
        img_height = img.height

        label_lines = []
        for _, row in img_annotations.iterrows():
            # Extract the coordinates
            region_attrs = eval(row["region_shape_attributes"])
            if (
                "x" not in region_attrs
                or "y" not in region_attrs
                or "width" not in region_attrs
                or "height" not in region_attrs
            ):
                print(
                    f"Error: 'x', 'y', 'width', or 'height' not found in region_shape_attributes for image {img_name}"
                )
                continue
            x = region_attrs["x"]
            y = region_attrs["y"]
            w = region_attrs["width"]
            h = region_attrs["height"]

            # Convert in YOLO  format (normalized)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            # Get the class
            class_id = eval(row["region_attributes"])
            if "Classes" in class_id:
                class_id = class_id["Classes"]
            else:
                print(
                    f"Error: 'Classes' not found in region_attributes for image {img_name}"
                )
                class_id = class_ids[class_id]
                continue

            try:
                class_id = int(class_id)
            except ValueError:
                print(f"Error: class_id is not an integer for image {img_name}")
                class_id = None
                continue
            class_ids.append(class_id)
            # Créer la ligne au format YOLO
            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # Write the label file
        label_path = join(output_dir, os.path.splitext(img_name)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    log_class_distribution(config, class_ids)


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


def log_class_distribution(config, class_ids):
    class_mapping = {str(i): class_name for i, class_name in enumerate(config["names"])}
    # Count occurrences of class_ids as values
    class_distribution = {class_name: 0 for class_name in class_mapping.values()}

    # Count occurrences of class_ids as keys
    for class_id in class_ids:
        if class_id in class_mapping:
            class_name = class_mapping[class_id]
            class_distribution[class_name] += 1

    # Count occurrences of class_ids as values
    for class_id, class_name in class_mapping.items():
        class_distribution[class_name] += class_ids.count(int(class_id))

    print(class_distribution)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str)
    args = parser.parse_args()

    df = pd.read_csv(
        join(args.path_data, "labels", "annotations.csv"),
        delimiter=",",
        engine="python",
        header=0,
    )
    print(df.head())
    with open(join(args.path_data, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    process_image_set(df, args.path_data, config)
