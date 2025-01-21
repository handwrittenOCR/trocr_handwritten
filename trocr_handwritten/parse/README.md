# Document Layout Annotation Guide

This guide explains how to annotate documents for layout parsing and prepare the data for training. The process involves several steps, from manual annotation to preparing the data for model training.

## Table of Contents
1. [Manual Annotation Process](#1-manual-annotation-process)
2. [Converting Annotations to YOLO Format](#2-converting-annotations-to-yolo-format)
3. [Preparing Dataset for Training](#3-preparing-dataset-for-training)
4. [Using Model Predictions for New Annotations](#4-using-model-predictions-for-new-annotations)

## 1. Manual Annotation Process

### Setup VGG Image Annotator (VIA)
1. Visit [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)
2. Use the web version or download the standalone application
3. Load your images into VIA

### Annotation Guidelines
1. Create the following attribute in VIA:
   - Name: "Classes"
   - Type: Dropdown
   - Options:
     - Title (0)
     - En-tête (1)
     - Marge (2)
     - Nom (3)
     - Plein Texte (4)
     - Signature (5)
     - Table (6)
     - Section (7)

2. Draw rectangles around each element in your documents
3. Select the appropriate class for each rectangle
4. Save your work regularly

### Export Annotations
1. Export annotations as CSV file
2. Name it `annotations.csv`

## 2. Converting Annotations to YOLO Format

### Prepare Your Directory Structure
```
path_data/
├── images/
│   └── (your image files in .jpg)
├── labels/
│   └── annotations.csv
└── config.yaml
```

The config file should look like this:
```yaml
path: ./data/  # root path
train: images/train  # training set for images
val: images/val      # validation set for images

# Classes
nc: 8
names: ['Titre', 'En-tête', 'Marge','Nom','Plein Texte', 'Signature', 'Table', 'Section']

```

### Convert Annotations
Run the conversion script:
```bash
python convert_to_yolo_format.py --path_data path/to/your/data
```

This will create YOLO format annotations (.txt files) in the labels directory.

## 3. Preparing Dataset for Training

### Required Directory Structure
After conversion, ensure your directory looks like this:
```
Yolo_annotated/
├── images/
│   └── (your .jpg files)
├── labels/
│   └── (your .txt files)
└── config.yaml
```

### Split and Push to Hugging Face
To split your dataset into train/test sets and push to Hugging Face Hub:

```bash
python split_and_push.py \
    --path_to_images path/to/Yolo_annotated/images \
    --path_to_labels path/to/Yolo_annotated/labels \
    --repo_name your-repo-name \
    --username your-username
```

This will:
- Split your dataset (80% train, 20% test)
- Create a balanced split
- Push the dataset to Hugging Face Hub

## 4. Using Model Predictions for New Annotations

If you want to use model predictions to speed up annotation of new documents:

1. Run inference on new images:
```bash
python layout_parser.py
```

2. The script will:
   - Generate predictions for your images
   - Create a JSON file compatible with VIA
   - Save structured crops of detected regions

3. In VIA:
   - Load the generated JSON file
   - Review and correct predictions
   - Export corrected annotations as CSV

4. Convert the corrected annotations to YOLO format using the process in Section 2

## Tips and Troubleshooting

- Ensure all image files use lowercase extensions (e.g., .jpg not .JPG)
- To convert image extensions to lowercase:
```bash
for f in *.JPG; do mv -- "$f" "${f%.JPG}.jpg"; done
```
- Keep backup copies of your annotations
- Verify the class distribution in your dataset after splitting
- Check the generated YOLO format files to ensure correct conversion

## Support

For technical issues or questions:
- Check the error messages in the console
- Verify your directory structure matches the requirements
- Ensure all file permissions are correct
