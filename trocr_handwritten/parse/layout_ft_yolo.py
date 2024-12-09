#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
from os.path import join
from doclayout_yolo import YOLOv10


def calculate_iou(box1, box2):
    """
    Calcule l'IoU entre deux boîtes.

    Args:
        box1: Boîte 1 (x, y, w, h)
        box2: Boîte 2 (x, y, w, h)
    Returns:
        float: IoU entre les deux boîtes
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculer les coordonnées des coins
    x1_1, y1_1, x1_2, y1_2 = x1, y1, x1 + w1, y1 + h1
    x2_1, y2_1, x2_2, y2_2 = x2, y2, x2 + w2, y2 + h2

    # Calculer l'intersection
    xi1, yi1 = max(x1_1, x2_1), max(y1_1, y2_1)
    xi2, yi2 = min(x1_2, x2_2), min(y1_2, y2_2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Calculer l'union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # Calculer l'IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def create_via_json(detection_results, class_names, iou=0.5):
    """
    Crée un fichier JSON au format VIA à partir des résultats de détection YOLO

    Args:
        detection_results: Liste des résultats de détection YOLO
        class_names: Dictionnaire de mapping des classes
        iou: Seuil d'intersection sur l'union pour les prédictions (default: 0.5)
    Returns:
        dict: JSON au format VIA
    """
    via_json = {
        "_via_settings": {
            "ui": {
                "annotation_editor_height": 25,
                "annotation_editor_fontsize": 0.8,
                "leftsidebar_width": 18,
                "image_grid": {
                    "img_height": 80,
                    "rshape_fill": "none",
                    "rshape_fill_opacity": 0.3,
                    "rshape_stroke": "yellow",
                    "rshape_stroke_width": 2,
                    "show_region_shape": True,
                    "show_image_policy": "all",
                },
                "image": {
                    "region_label": "Classes",
                    "region_color": "__via_default_region_color__",
                    "region_label_font": "10px Sans",
                    "on_image_annotation_editor_placement": "NEAR_REGION",
                },
            },
            "core": {"buffer_size": 18, "filepath": {}, "default_filepath": ""},
            "project": {"name": "yolo_predictions"},
        },
        "_via_attributes": {
            "region": {
                "Classes": {
                    "type": "dropdown",
                    "description": "",
                    "options": class_names,
                    "default_options": {},
                }
            },
            "file": {},
        },
        "_via_data_format_version": "2.0.10",
        "_via_img_metadata": {},
        "_via_image_id_list": [],
    }

    for res in detection_results:
        filename = Path(res.path).name
        filesize = Path(res.path).stat().st_size
        image_id = f"{filename}{filesize}"

        regions = []
        boxes = res.__dict__["boxes"].xywh
        classes = res.__dict__["boxes"].cls

        for box, cls in zip(boxes, classes):
            x, y, w, h = box.tolist()
            class_id = int(cls.item())

            new_box = [int(x - w / 2), int(y - h / 2), int(w), int(h)]
            is_duplicate = False

            for existing_region in regions:
                existing_box = [
                    existing_region["shape_attributes"]["x"],
                    existing_region["shape_attributes"]["y"],
                    existing_region["shape_attributes"]["width"],
                    existing_region["shape_attributes"]["height"],
                ]
                if calculate_iou(new_box, existing_box) > iou:
                    is_duplicate = True
                    break

            if not is_duplicate:
                region = {
                    "shape_attributes": {
                        "name": "rect",
                        "x": new_box[0],
                        "y": new_box[1],
                        "width": new_box[2],
                        "height": new_box[3],
                    },
                    "region_attributes": {"Classes": str(class_id)},
                }
                regions.append(region)

        image_metadata = {
            "filename": filename,
            "size": filesize,
            "regions": regions,
            "file_attributes": {},
        }

        via_json["_via_img_metadata"][image_id] = image_metadata
        via_json["_via_image_id_list"].append(image_id)

    return via_json


def process_folder(path_folder, path_model, conf=0.2, iou=0.5):
    """
    Traite un dossier d'images avec le modèle YOLO et génère un fichier JSON VIA

    Args:
        path_folder: Chemin vers le dossier contenant les images
        path_model: Chemin vers le fichier modèle .pt
        conf: Seuil de confiance pour les prédictions (default: 0.2)
        iou: Seuil d'intersection sur l'union pour les prédictions (default: 0.5)
    """
    # Vérifier que les chemins existent
    if not Path(path_folder).exists():
        raise ValueError(f"Le dossier {path_folder} n'existe pas")
    if not Path(path_model).exists():
        raise ValueError(f"Le modèle {path_model} n'existe pas")

    # Charger le modèle
    print(f"Chargement du modèle {path_model}...")
    model = YOLOv10(path_model)

    # Faire les prédictions
    print(f"Prédiction sur les images dans {path_folder}...")
    det_res = model.predict(path_folder, imgsz=1024, conf=conf, iou=iou, device="cpu")

    # Définir les noms des classes
    class_names = {
        "0": "Title",
        "1": "En-tête",
        "2": "Marge",
        "3": "Nom",
        "4": "Plein Texte",
        "5": "Signature",
        "6": "Table",
    }

    # Créer le JSON
    print("Création du fichier JSON...")
    via_json = create_via_json(det_res, class_names, iou)

    # Sauvegarder le JSON
    output_path = join(path_folder, "yolo_predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(via_json, f, ensure_ascii=False, indent=2)

    print(f"Fichier JSON sauvegardé : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Applique un modèle YOLO sur un dossier d'images et génère un fichier JSON VIA"
    )
    parser.add_argument(
        "--path_folder",
        required=True,
        help="Chemin vers le dossier contenant les images",
    )
    parser.add_argument(
        "--path_model", required=True, help="Chemin vers le fichier modèle .pt"
    )
    parser.add_argument(
        "--conf", type=float, default=0.2, help="Seuil de confiance (entre 0 et 1)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="Seuil d'intersection sur l'union (entre 0 et 1)",
    )

    args = parser.parse_args()

    # Vérifier la valeur de conf
    if not 0 <= args.conf <= 1:
        raise ValueError("La valeur de conf doit être entre 0 et 1")

    process_folder(args.path_folder, args.path_model, args.conf, args.iou)


if __name__ == "__main__":
    main()
