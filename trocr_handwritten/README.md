# Pipeline Components

In this page we explain a bit more in detail the different components of the pipeline. You can test it by yourself on the exemple provided in `data` folder.

# Table of Contents

1. [Installation](#-installation)
2. [Parsing Layout](#-parsing-layout)
3. [Handwritten Optical Character Recognition (OCR)](#-optical-character-recognition-ocr)
   - [Training](#-training-a-custom-trocr-model)
   - [Applying](#-applying-trocr)
4. [Named Entity Recognition (NER)](#-named-entity-recognition-ner-with-llm)


## 📦 Installation

### Step 1: Clone the Repository

Start by cloning the GitHub repository to your local machine. Open a terminal and run the following command:

```bash
git clone https://github.com/handwrittenOCR/trocr_handwritten.git
```

### Step 2: Install Poetry
`Poetry` is a tool for dependency management in Python. Install it with the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Step 3: Install Dependencies

Navigate to the `trocr_handwritten` directory and install the dependencies with the following command:

```bash
poetry install
```

NB: If you have Python 3.10 installed, you can use the following command to install the dependencies:

```bash
sudo apt update
sudo apt install python3.11
poetry env use python3.11
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

### 🧪 Applying layout parser

#### 🎯 Objective

The objective of this script is to process historical document images and parse their layout using a fine-tuned YOLOv10 model. The script identifies different document elements (such as titles, headers, margins, names, text blocks, signatures, and tables) and can output both structured cropped images and VIA-format annotations. The model goes much faster (~x10) on a GPU (1s/image on CPU vs 0.1s/image on GPU A10).

#### 🛠️ How it works

The script works in the following steps:

1. **Model Loading**: The script loads a YOLOv10 model either from a local file or from the HuggingFace Hub.

2. **Layout Detection**: The model processes input images and detects various document elements, assigning them to predefined classes:
   - Title
   - En-tête (Header)
   - Marge (Margin)
   - Nom (Name)
   - Plein Texte (Full Text)
   - Signature
   - Table

3. **Output Generation**: The script can generate two types of outputs:
   - Structured cropped images organized by document element type
   - VIA-format JSON annotations for further processing or visualization (optional)

#### 📜 How to apply the script

To apply the script, you need to have a set of images of historical documents in a folder.

1. **Set the settings**

You should configure the script using the `LayoutParserSettings` class with the following parameters in `trocr_handwritten/parse/settings.py`:

- `path_folder`: Path to input images (default: "data/raw/images")
- `path_output`: Path for processed outputs (default: "data/processed/images/")
- `path_model`: Path to local model file (optional if None you should have a model in the HuggingFace Hub)
- `hf_repo`: HuggingFace repository name (default: "agomberto/historical-layout-ft")
- `hf_filename`: Model filename in HF repo (default: "20241119_v2_yolov10_50_finetuned.pt")
- `device`: Computing device ("cpu" or "cuda")
- `conf`: Confidence threshold (default: 0.2)
- `iou`: Intersection over Union threshold (default: 0.5)
- `create_annotation_json`: Whether to create a file for [VIA annotations](https://annotate.officialstatistics.org/) (default: True)

2. **Run the script**

```bash
python trocr_handwritten/parse/parse_page.py
```

The final directory structure will be:

```bash
|-data
  |-raw
     |-images
        |-document1.jpg
        |-document2.jpg
  |-processed
     |-images
        |-document1
           |-Title
              |-000.jpg
           |-Plein Texte
              |-000.jpg
              |-001.jpg
           |-metadata.json
```

#### On Notebooks

Once again, be sure to set the settings in the `LayoutParserSettings` class in `trocr_handwritten/parse/settings.py`.
Example usage:

```python
from trocr_handwritten.parse.settings import LayoutParserSettings
from trocr_handwritten.parse.utils import YOLOv10Model

settings = LayoutParserSettings()
model = YOLOv10Model(settings, logger)
results = model.predict(settings.path_folder)
```

The script will process all images in the input folder and create:
1. A structured folder system containing cropped images for each detected element
2. A metadata.json file for each processed document containing coordinates and classifications
3. (Optional) VIA-format annotations for visualization and further processing

### 🚅 Training a customed layout parser

The layout parser can be fine-tuned on your own dataset using our provided scripts. We'll guide you through the process using cloud GPU resources (specifically Lambda Labs A10/A100).

#### Architecture Overview

Our fine-tuning pipeline consists of several components:

1. **Environment Setup** (`install_doclayout.sh`):
   - Installs Miniconda and creates a dedicated environment (it uses python 3.10 instead of 3.11 that we use in the rest of the project)
   - Clones and installs DocLayout-YOLO dependencies
   - Sets up necessary Python packages
   - Connects to HuggingFace hub

2. **Data Preparation** (`prepare_data.sh`):
   - Downloads dataset from Hugging Face Hub
   - Creates YOLO-format directory structure
   - Splits images and labels into train/val sets
   - Downloads configuration and pretrained model

3. **Training and Model Publishing** (`train_yolo.sh`):
   - Configures training parameters
   - Runs training process with specified hyperparameters
   - Saves best model checkpoint
   - Uploads trained model to Hugging Face Hub
   - Handles versioning and metadata

#### Step-by-Step Guide

1. **Install Doclayout Environment**
```bash
cd trocr_handwritten/parse
chmod +x *.sh
./install_doclayout.sh
```

2. **Prepare Data**
```bash
conda activate doclayout_yolo # you may meed to restart your termial
./prepare_data.sh
```

3. **Training and Push Model**
```bash
./train_yolo.sh
```

#### Using Lambda Labs

To run this on Lambda Labs A10/A100:

1. **Create Instance**:
   - Visit [Lambda Labs Console](https://cloud.lambdalabs.com)
   - Select A10 (24GB) or A100 (40GB/80GB) instance
   - Choose Ubuntu 20.04 image

2. **SSH to Instance**:
```bash
ssh ubuntu@<instance-ip>
```

3. **Clone and Setup**:
```bash
git clone https://github.com/agomberto/trocr_handwritten.git
cd trocr_handwritten/parse
./install_doclayout.sh
```

4. **Training Configuration**:
For A10 (24GB), recommended settings in `train_yolo.sh`:
```bash
BATCH_SIZE=8
IMAGE_SIZE=1024
```

For A100 (80GB), you can increase to:
```bash
BATCH_SIZE=16
IMAGE_SIZE=1024
```

5. **Monitor Training**:
- Training progress is logged in `yolo_ft/`
- Use `nvidia-smi` to monitor GPU usage
- Results will be saved in `yolo_ft/best.pt`

#### Training Parameters

Key parameters in `train_yolo.sh`:
```bash
EPOCHS=50        # Number of training epochs
PATIENCE=5       # Early stopping patience
BATCH_SIZE=8     # Batch size per GPU
LR=0.001         # Initial learning rate
```

Adjust these based on your GPU memory and dataset size. For the A100, you can typically use larger batch sizes and potentially increase the image size.

#### Expected Results

After training, you should see:
- Training metrics in `yolo_ft/`
- Best model saved as `yolo_ft/best.pt`
- Validation results including mAP scores

The model can then be pushed to the Hugging Face Hub using the provided script for easy sharing and versioning.

## 🔎 Handwritten Optical Character Recognition (OCR)

### 🚅 Training a custom TrOCR model

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

NB: You can modify the configuration files to change the model parameters and training arguments. Train it on GPUs if possible and then you can change the parameter `FP16`to `True` to use mixed precision training.

Example command:

```bash
python trocr_handwritten/trocr/train_trocr.py --PATH_CONFIG /path/to/config --dataset /path/to/dataset --processor /path/to/processor --trocr_model /path/to/model
```

### 🧪 Applying TrOCR

#### 🎯 Objective

The objective of this script is to apply a trained Optical Character Recognition (OCR) model to transcribe handwritten text from images (in lines). The script uses the TrOCR model architecture, which is a combination of Vision and Language models, to recognize and transcribe the handwritten text. The script takes as input a set of images, processes them using the TrOCR model, and outputs the transcribed text to a specified file. The model used can be specified by the user, allowing for flexibility in the transcription process.

#### 🛠️ How it works

The script works in the following steps:

1. **Model and Processor Loading**: The trained TrOCR processor and model, and the tokenizer are loaded.

2. **Device Selection**: The script checks if a GPU is available and if so, moves the model to the GPU.

3. **Data Loading and Processing**: The script loads and processes the images from the provided directory. It converts the images to RGB and processes them using the TrOCR processor.

4. **Text Generation**: The script generates text from the processed images using the trained TrOCR model.

5. **Output Writing**: The script writes the generated text to the specified output file.


#### 📜 How to apply the script

To apply the script, you need to have a set of images of handwritten texts, and optionally a pre-trained TrOCR model and a specific processor.

You can run the script from the command line with the following arguments:

- `--PATH_DATA`: Path to the directory containing the images.
- `--trocr_model`: Path to the trained TrOCR model. Default is "agomberto/trocr-base-handwritten-fr".
- `--processor`: Path to the processor. Default is "microsoft/trocr-large-handwritten".
- `--PATH_OUTPUT`: Path to the output file where the transcriptions will be written.

Example command:

```bash
python trocr_handwritten/trocr/apply_trocr.py --PATH_DATA /path/to/data --trocr_model /path/to/model --processor /path/to/processor --PATH_OUTPUT /path/to/output
```

The script will load the specified model and processor, load and process the images, generate text from the images using the model, and write the generated text to the specified output file. The results of the transcription will be logged and can be viewed in the console.

If we take back the example of recommanded structure:

The recomanded structure would be to have
```bash
|-data
  |-pages
     |-page_1.jpg
  |-xml
     |-page_1.xml
  |-lines
      |-page_1
          |-column_0
            |-page_1_line_0.jpg
            |-page_1_line_1.jpg
          |-column_1
            |-page_1_line_0.jpg
            |-page_1_line_1.jpg
            |-page_1_line_2.jpg
```

The final structure will be

```bash
|-data
  |-pages
     |-page_1.jpg
  |-xml
     |-page_1.xml
  |-lines
      |-page_1
          |-column_0
            |-page_1_line_0.jpg
            |-page_1_line_1.jpg
          |-column_1
            |-page_1_line_0.jpg
            |-page_1_line_1.jpg
            |-page_1_line_2.jpg
  |-ocrized
      |-page_1
          |-column_0
            |-ocrized.txt
          |-column_1
            |-ocrized.txt
```

## 🗃️ Named Entity Recognition (NER) with LLM

### 🎯 Objective

The objective of this script is to apply a Named Entity Recognition (NER) schema using a Large Language Model (LLM). The script takes as input a text file, applies a template to it, and sends it to the OpenAI API for processing. The API response is then printed. The script uses the OpenAI Python client and the Jinja2 templating engine. It also allows for customization of the input text, template, and schema files through command-line arguments.

### 🛠️ How it works

The script works in the following steps:

1. **Text Loading and Processing**: The script loads the text from the specified file, removes parentheses, and applies a Jinja2 template to it.

2. **Schema Loading**: The script loads the schema from the specified JSON file. Schema by the LLM to output the desired entities in the desired format.

3. **API Request**: The script sends a request to the OpenAI API, using the processed text and the loaded schema. The request specifies the use of the "gpt-3.5-turbo" model, and includes system and user messages.

4. **Output Printing**: The script prints the arguments of the function call from the API response: the dictionary of entities.

### 📜 How to apply the script

To apply the script, you need to have a text file and a specific prompt and schema.

You can run the script from the command line with the following arguments:

- `--PATH_DATA`: Path to the directory containing the text file.
- `--text`: Name of the text file (without extension). Default is "example_birth_act".
- `--PATH_CONFIG`: Path to the directory containing the config file, schema, and prompt.
- `--prompt`: Name of the prompt file (without extension). Default is "birth_act".
- `--schema`: Name of the schema file (without extension). Default is "birth_act_schema".

Example command:

```bash
export OPENAI_API_KEY='your-api-key'
python ner_GPT.py --PATH_DATA /path/to/data --text example_birth_act --PATH_CONFIG /path/to/config --prompt birth_act --schema birth_act_schema
```

Here is the output you should get for the birth act.

```json
{
  "name": "Justine Laurence",
  "sex": "femme",
  "birth date": "2 Mai",
  "birth place": "habitation Douillard section de Caraque",
  "father's name": "Clotilde Pierre Auguste",
  "father's age": "29",
  "father's job": "cultivateur",
  "father's birth place": "None",
  "father's residence place": "Aux Abymes",
  "mother's name": "Au can Marie Augustine",
  "mother's age": "24",
  "mother's job": "couturière",
  "mother's birth place": "None",
  "mother's residence place": "Aux Abymes"
}
```
