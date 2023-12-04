# Pipeline Components

In this page we explain a bit more in detail the different components of the pipeline. You can test it by yourself on the exemple provided in `data` folder.

# Table of Contents

1. [Installation](#-installation)
2. [Parsing Layout](#-parsing-layout)
3. [Handwritten Optical Character Recognition (OCR)](#-optical-character-recognition-ocr)
   - [Training](#-training)
   - [Applying](#-applying)
4. [Named Entity Recognition (NER)](#-named-entity-recognition-ner)


## 📦 Installation

### Step 1: Clone the Repository

Start by cloning the GitHub repository to your local machine. Open a terminal and run the following command:

```bash
git clone https://github.com/handwrittenOCR/trocr_handwritten.git
```

### Step 2: Install Poetry
`Poetry` is a tool for dependency management in Python. Install it with the following command:

```bash
curl -sSL https://install.python-poetry.org | python -
```

### Step 3: Install Dependencies

Navigate to the `trocr_handwritten` directory and install the dependencies with the following command:

```bash
poetry install
```

### Step 4: Install Pre-commit Hooks

Pre-commit hooks help to catch issues before code is committed to the repository. Install the pre-commit package and set up the pre-commit hooks with the following commands:

```bash
poetry run pre-commit install
```

### Step 5: Activate the environment

Activate the Poetry-managed virtual environment before working:

```bash
poetry shell
```


## 🔬 Parsing Layout

### 🎯 Objective

The objective of this script is to process images and parse it in columns and lines using a model trained with the ARU-Net architecture. The script takes as input a set of images, generate from each page an XML file then processes the XML file to extract bounding boxes (bboxes), and saves these bboxes as separate images. It also provides an option to visualize the bboxes on the original images.

### 🛠️ How it works

The script works in the following steps:

1. **Configuration**: The script reads a configuration file (`config.json`) to set up the parameters for the ARU-Net model and other processing steps.

2. **Model Setup**: The ARU-Net model is created and set up using the parameters from the configuration file.

3. **Data Loading**: The script loads the images to be processed.

4. **Model Application**: The script applies the ARU-Net model to the loaded images and saves the corresponding XML files that identify lines of texts.

5. **Bbox Extraction**: The script parses the XML files to extract columns and in each column the bbox coordinates. It then resizes and splits the bboxes according to the column they belong to.

6. **Bbox Saving**: Each bbox is cropped from the original image and saved as a separate image in a specified directory. The bboxes are organized by the column they belong to.

7. **Visualization**: If the verbose option is enabled, the script displays the original images with the bboxes drawn on them.

### 📜 How to apply the script

To apply the script, you need to have a set of images and corresponding XML files containing bbox information. You also need to have a trained ARU-Net model saved as a `.tar` file.

You can run the script from the command line with the following arguments:

- `--PATH_PAGES`: Path to the directory containing the images.
- `--PATH_MODELS`: Path to the directory containing the model file.
- `--PATH_XML`: Path to the directory containing the XML files.
- `--PATH_LINES`: Path to the directory where the bbox images will be saved.
- `--verbose`: (Optional) If set to `True`, the script will display the images with the bboxes drawn on them.

Example command:

```bash
python trocr_handwritten/parse/parse_page.py --PATH_PAGES /path/to/images --PATH_MODELS /path/to/model --PATH_XML /path/to/xml --PATH_LINES /path/to/save/bbox/images --verbose True
```

## 🔎 Handwritten Optical Character Recognition (OCR)

### 🚅 Training

#### 🎯 Objective

The objective of this script is to train a model for Optical Character Recognition (OCR) on handwritten text. The script uses the TrOCR model architecture, which is a combination of Vision and Language models, to recognize and transcribe the handwritten text from images. The script takes as input a set of images, processes them using the TrOCR model, and outputs the transcribed text. The model is trained using a dataset of handwritten texts and the performance is evaluated using Character Error Rate (CER) and Word Error Rate (WER) metrics.

#### 🛠️ How it works

The script works in the following steps:

1. **Model and Processor Loading**: The TrOCR processor and model, and the tokenizer are loaded.

2. **Data Loading and Processing**: The script loads and processes the data using the provided dataset, tokenizer, and processor. It uses by default a HandwrittenTextDataset class from the `datasets` library called French Census.

3. **Model & Training Configuration**: The script configures the model parameters & training argument using the provided configuration files.

4. **Metric Loading**: The script loads the Character Error Rate (CER) and Word Error Rate (WER) metrics for evaluation.

5. **Trainer Setup**: The script sets up the trainer with the model, tokenizer, training arguments, metrics, and data.

6. **Model Training**: The script trains the model using the configured trainer.


#### 📜 How to apply the script

To apply the script, you need to have a dataset of handwritten texts, a configuration file, and optionally a pre-trained TrOCR model.

You can run the script from the command line with the following arguments:

- `--PATH_CONFIG`: Path to the configuration files.
- `--dataset`: Path to the dataset. Default is "agomberto/FrenchCensus-handwritten-texts".
- `--processor`: Path to the processor. Default is "microsoft/trocr-large-handwritten".
- `--trocr_model`: Path to the pre-trained TrOCR model. Default is "microsoft/trocr-large-handwritten".

Example command:

```bash
python trocr_handwritten/parse/train_trocr.py --PATH_CONFIG /path/to/config --dataset /path/to/dataset --processor /path/to/processor --trocr_model /path/to/model
```

### 🧪 Testing




## 🗃️ Named Entity Recognition (NER)
