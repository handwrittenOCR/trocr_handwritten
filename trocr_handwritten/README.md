# Pipeline Components

In this page we explain a bit more in detail the different components of the pipeline. You can test it by yourself on the exemple provided in `data` folder.

# Table of Contents

1. [Installation](#-installation)
2. [Parsing Layout](#-parsing-layout)
   - [Applying](#-applying-layout-parser)
   - [Training](#-training-a-custom-layout-parser)
3. [Line Segmentation](#-line-segmentation)
   - [Applying](#-applying-line-segmentation)
4. [Handwritten Optical Character Recognition (OCR)](#-optical-character-recognition-ocr)
   - [Training](#-training-a-custom-trocr-model)
   - [Applying](#-applying-trocr)
5. [Named Entity Recognition (NER)](#-named-entity-recognition-ner-with-llm)


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
# Install main dependencies without kraken
poetry install
```

NB: If you have Python 3.10 installed, you can use the following command to install the dependencies:

```bash
sudo apt update
sudo apt install python3.11
poetry env use python3.11
# poetry install
poetry install
```

Note: The kraken package for line segmentation has specific dependency requirements. It will be installed separately when needed. See the [Line Segmentation](#-line-segmentation) section for details.

### Step 4: Install Pre-commit Hooks

Pre-commit hooks help to catch issues before code is committed to the repository. Install the pre-commit package and set up the pre-commit hooks with the following commands:

```bash
poetry run pre-commit install
```

### Step 5: Activate the environment

Activate the Poetry-managed virtual environment before working:

```bash
poetry self add poetry-plugin-shell
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
   - Header
   - Margin
   - Name
   - Full Text
   - Signature
   - Table
   - Section
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
python trocr_handwritten/parse/layout_parser.py
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

2. **Prepare Data, Train and Push Model**
```bash
source ~/.bashrc
conda activate doclayout_yolo
cd DocLayout-YOLO
chmod +x *.sh
./prepare_data_and_train.sh
```

NB: you may have to modify the `~/.config/Ultralytics/settings.yaml` file to set the good path to the dataset and then run the script again.

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

## 🔬 Line Segmentation

### 🧪 Applying line segmentation

#### 🎯 Objective

The line segmentation component uses Kraken to detect and extract individual lines of text from document images. It handles both margin text and main text differently, with specific parameters for each type.

#### 🛠️ How it works

The script works in parallel using asyncio and ThreadPoolExecutor to process multiple images efficiently:

1. **Model Loading**: Uses Kraken's blla model for line detection
2. **Parallel Processing**: Processes multiple images simultaneously using worker pools
3. **Different Parameters**:
    - Margins: Uses width padding (50px) and no IoU filtering
    - Main Text: Uses height padding (15px) and IoU filtering (0.5)
4. **Output Generation**: Creates structured output with:
    - Cropped line images
    - Metadata JSON files containing coordinates and paths

#### 📜 How to apply the script

First, set up the Kraken environment:

```bash
# Navigate to the kraken environment directory
cd trocr_handwritten/segmentation/kraken_env

# Install dependencies
poetry install

# Run the segmentation script
poetry run python -m line_segmenter.run_segmentation
```

The script will:
1. Download the Kraken model if needed
2. Process all images in parallel
3. Generate logs and timing information
4. Create a structured output directory

The output structure will be:

```bash
data/
├── processed/
│   └── images/
│       └── document_folder/
│           ├── Marge/
│           │   └── image_name/
│           │       ├── lines/
│           │       │   ├── line_1.jpg
│           │       │   └── line_2.jpg
│           │       └── metadata.json
│           └── Plein Texte/
│               └── image_name/
│                   ├── lines/
│                   │   ├── line_1.jpg
│                   │   └── line_2.jpg
│                   └── metadata.json
└── logs/
    └── segmentation_YYYYMMDD_HHMMSS.log
```

You can also use the provided Jupyter notebook for interactive processing and visualization:

```bash
# Start Jupyter notebook
poetry run jupyter notebook notebooks/line_segmentation.ipynb
```

### 📦 Data Management with AWS S3

The project includes a FileManager utility for handling data transfers between local storage and AWS S3:

First, create a `.env` file in the project root with your AWS credentials:

```bash
# .env
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
```

```python
from trocr_handwritten.utils.file_manager import FileManager, S3Config

# Configure S3 access
s3_config = S3Config(
    bucket_name="your-bucket",
)

# Initialize file manager
file_manager = FileManager(
    local_path="data/processed",
    s3_config=s3_config
)

# Download data from S3
file_manager.download_from_s3(
    s3_prefix="raw/images",
    local_subdir="images"
)

# Upload processed data to S3
file_manager.upload_to_s3(
    local_dir="data/processed/images",
    s3_prefix="processed/images",
    include_patterns=["*.jpg", "*.json"]
)

# Clean local directory when done
file_manager.clean_local_directory("images")
```

You can also use the command-line interface:

```bash
# Download from S3
s3-manage download --bucket your-bucket --s3-prefix raw/images --local-dir data/processed

# Upload to S3
s3-manage upload --bucket your-bucket --s3-prefix processed/images --local-dir data/processed/images

# Clean local directory
s3-manage clean --local-dir data/processed --subdir images
```

Run `s3-manage --help` for more information about available commands and options.

## 🔎 Handwritten Optical Character Recognition (OCR)

### 🚅 Training a custom TrOCR model

#### 🎯 Objective

The TrOCR training pipeline combines Vision and Language models to recognize and transcribe handwritten text from images. The model's performance is evaluated using Character Error Rate (CER) and Word Error Rate (WER) metrics.

#### 🛠️ Components

1. **Configuration Management**:
   - Uses Pydantic models for type-safe configuration
   - Supports both model and training settings
   - Handles Hugging Face Hub integration

2. **Dataset Handling**:
   - Processes both census and private datasets
   - Supports automatic train/validation/test splitting
   - Implements custom data collation for TrOCR inputs

3. **Model Training**:
   - Uses a custom TrOCR trainer for proper loss computation
   - Supports mixed precision training (FP16)
   - Implements early stopping and model checkpointing
   - Provides comprehensive logging and metrics tracking

#### 📜 Training Steps

1. **Configure Settings**

Create a training script or use the provided `train.py`:

```python
from trocr_handwritten.trocr.settings import (
    TrainSettings,
    OCRModelSettings,
    TrainerDatasetsSettings
)

# Model settings
model_settings = OCRModelSettings(
    model_name="microsoft/trocr-large-handwritten",
    hub_repo=None  # Set if loading from Hub
)

# Training settings
train_settings = TrainSettings(
    output_dir="./results",
    push_to_hub=True,
    hub_model_id="your-model-name",
    private_hub_repo=True,
    training_config=TrainingConfig(
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=4e-5,
        num_train_epochs=20,
        fp16=True
    )
)

# Dataset settings
dataset_settings = TrainerDatasetsSettings(
    census_data=True,
    private_repo="your-private-dataset",  # Optional
    max_label_length=64
)
```

2. **Train the Model**

```python
from trocr_handwritten.trocr.model import OCRModel
from trocr_handwritten.trocr.dataset import TrainerDatasets

# Initialize model and datasets
ocr_model = OCRModel(settings=model_settings, train_settings=train_settings)
trainer_datasets = TrainerDatasets(
    settings=dataset_settings,
    tokenizer=ocr_model.tokenizer,
    processor=ocr_model.processor
)

# Load data and train
datasets = trainer_datasets.load_and_process_data()
compute_metrics_fn = OCRModel.setup_compute_metrics(
    ocr_model.tokenizer,
    ocr_model.processor
)

result, trainer = ocr_model.train(
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    compute_metrics_fn=compute_metrics_fn
)
```

3. **Push to Hugging Face Hub**

```python
ocr_model.push_to_hub(
    repo_name="your-model-name",
    huggingface_api_key="your-api-key",
    private=True,
    metrics=test_metrics
)
```

### 📊 Evaluating a TrOCR Model

The project provides two ways to evaluate a TrOCR model:

1. **Using evaluate_trocr.py**

```bash
python trocr_handwritten/trocr/evaluate_trocr.py \
    --model_name "your-model-name" \
    --census_data \
    --private_repo "your-private-dataset" \
    --max_label_length 64
```

This will:
- Load the specified model
- Evaluate it on test datasets
- Output metrics including CER and WER
- Generate detailed evaluation reports

2. **Using example_trocr_apply.py**

```bash
python trocr_handwritten/trocr/example_trocr_apply.py \
    --model_name "your-model-name" \
    --num_samples 20 \
    --seed 42 \
    --census_data
```

This will:
- Generate predictions on random samples
- Create visualizations of the results
- Save a PDF with side-by-side comparisons
- Help qualitatively assess model performance

#### 🔍 Understanding Metrics

The evaluation provides several key metrics:

- **Character Error Rate (CER)**: Measures character-level differences between predictions and ground truth
- **Word Error Rate (WER)**: Measures word-level differences
- **Loss Values**: Training and validation losses
- **Prediction Examples**: Sample outputs with ground truth comparisons

#### 💾 Saving and Loading Models

Models can be saved locally or pushed to the Hugging Face Hub:

```python
# Save locally
trainer.save_model("path/to/save")

# Push to Hub
ocr_model.push_to_hub(
    repo_name="your-model-name",
    private=True,
    metrics=test_metrics
)

# Load from Hub
loaded_model = OCRModel(
    settings=OCRModelSettings(hub_repo="your-model-name"),
    train_settings=train_settings
)
```

#### 🔧 Best Practices

1. **Data Preparation**:
   - Clean and preprocess images consistently
   - Ensure balanced dataset splits
   - Validate data quality before training

2. **Training Configuration**:
   - Start with recommended hyperparameters
   - Use mixed precision (FP16) for faster training
   - Monitor validation metrics for early stopping

3. **Evaluation**:
   - Use both quantitative metrics and qualitative analysis
   - Test on diverse samples
   - Compare with baseline models

4. **Model Management**:
   - Version your models properly
   - Document training configurations
   - Keep evaluation results for comparison

### 🧪 Applying TrOCR

#### 🎯 Objective

The objective of this script is to apply a trained Optical Character Recognition (OCR) model to transcribe handwritten text from images (in lines). The script uses the TrOCR model architecture, which is a combination of Vision and Language models, to recognize and transcribe the handwritten text. The script takes as input a set of images, processes them using the TrOCR model, and outputs the transcribed text to a specified file. The model used can be specified by the user, allowing for flexibility in the transcription process.The project offers two approaches:

1. **Batch Processing (`apply_trocr.py`)**: Processes folders containing line images in batch mode, saving transcriptions to text files.
2. **Sample Evaluation (`example_trocr_apply.py`)**: Visualizes random samples from a test dataset with their predicted transcriptions.

#### 🛠️ How the Batch Processor Works

The batch processor (`apply_trocr.py`) works in the following steps:

1. **Recursive Folder Search**: Finds all "lines" folders in the provided directory structure.
2. **Model Loading**: Loads the specified TrOCR model, processor, and tokenizer.
3. **Batch Processing**: Processes images in batches to optimize performance.
4. **Transcription Generation**: Generates text transcriptions for each line image.
5. **Output Organization**: Saves transcriptions to structured text files.

#### 📜 How to Apply the Batch Processor

To transcribe multiple line images:

```bash
python trocr_handwritten/trocr/apply_trocr.py \
    --root_dir "data/processed/images" \
    --model_name "microsoft/trocr-large-handwritten" \
    --batch_size 32
```

Optional arguments:
- `--hf_token`: HuggingFace token for private models
- `--batch_size`: Number of images to process at once (default: 32)

The script will:
1. Find all "lines" folders recursively in the root directory
2. Process images in batches
3. Save transcriptions to "transcriptions.txt" files beside each lines folder

#### 🔍 Sample Evaluation and Visualization

The `example_trocr_apply.py` script provides a way to visualize model performance on random samples:

```bash
python trocr_handwritten/trocr/example_trocr_apply.py \
    --model_name "your-model-name" \
    --num_samples 20 \
    --seed 42 \
    --census_data
```

This will:
- Load the specified model
- Select random samples from the test dataset
- Generate predictions for each sample
- Create a visualization showing images with their predicted text
- Save the visualization as a PNG file

Optional arguments:
- `--num_samples`: Number of random samples to process (default: 20)
- `--seed`: Random seed for reproducibility (default: 42)
- `--census_data`: Whether to use census data for evaluation
- `--private_repo`: Private dataset repository
- `--max_label_length`: Maximum label length (default: 64)

#### 🔄 Directory Structure

For the batch processor, the expected input structure is:

```bash
data/
└── processed/
└── images/
└── document_folder/
   ├── lines/
      │ ├── line_1.jpg
      │ └── line_2.jpg
   └── Nom/
      └── lines/
         ├── line_1.jpg
         └── line_2.jpg
```

After processing, the structure will include:

```bash
data/
└── processed/
└── images/
└── document_folder/
   ├── lines/
      │ ├── line_1.jpg
      │ └── line_2.jpg
      ├── transcriptions.txt
   └── Nom/
      └── lines/
         ├── line_1.jpg
         └── line_2.jpg
        └── transcriptions.txt
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
