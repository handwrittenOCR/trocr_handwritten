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

### Step 2: Install uv
`uv` is a fast Python package manager. Install it with the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3: Install Dependencies

Navigate to the `trocr_handwritten` directory and install the dependencies with the following command:

```bash
uv sync
```

NB: If you need a specific Python version:

```bash
uv venv --python 3.11
uv sync
```

Note: The kraken package for line segmentation has specific dependency requirements. It will be installed separately when needed. See the [Line Segmentation](#-line-segmentation) section for details.

### Step 4: Install Pre-commit Hooks

Pre-commit hooks help to catch issues before code is committed to the repository. Install the pre-commit package and set up the pre-commit hooks with the following commands:

```bash
uv run pre-commit install
```

### Step 5: Activate the environment

Activate the virtual environment before working:

```bash
source .venv/bin/activate
```

## 🔬 Parsing Layout

### 🧪 Applying layout parser

#### 🎯 Objective

The layout parser detects document regions using a YOLO model. It supports both legacy YOLOv10 models (via `doclayout-yolo`) and new YOLO11 models (via `ultralytics`). The backend is auto-detected from the checkpoint. The model runs ~x10 faster on GPU (0.1s/image on A10 vs 1s/image on CPU).

#### Detected classes

| ID | Class |
|----|-------|
| 0 | Title |
| 1 | En-tête |
| 2 | Marge |
| 3 | Nom |
| 4 | Plein Texte |
| 5 | Signature |
| 6 | Table |
| 7 | Section |

#### 📜 Applying the layout parser

Configure via `LayoutParserSettings` in `trocr_handwritten/parse/settings.py`:

- `path_folder`: Input images directory (default: `data/raw/images`)
- `path_output`: Output directory (default: `data/processed/images/`)
- `path_model`: Local `.pt` model path (optional, uses HF if None)
- `hf_repo` / `hf_filename`: HuggingFace model repo and filename
- `device`: `"cpu"`, `"cuda"`, or `"mps"`
- `conf`: Confidence threshold (default: 0.2)
- `iou`: IoU threshold (default: 0.5)

```bash
python -m trocr_handwritten.parse.layout_parser
```

Output structure:

```
data/processed/images/
  document1/
    Title/
      000.jpg
    Plein Texte/
      000.jpg
      001.jpg
    metadata.json
```

In a notebook:

```python
from trocr_handwritten.parse.settings import LayoutParserSettings
from trocr_handwritten.parse.utils import YOLOModel

settings = LayoutParserSettings()
model = YOLOModel(settings, logger)
results = model.predict(settings.path_folder)
```

### 🏷️ Annotating data

The built-in annotation tool launches a local HTML server where you can draw, move, resize and delete bounding boxes on your images. Each annotation is randomly assigned to a split (70% train, 20% test, 10% dev) and saved to `data/layout/{split}/annotations.json`.

The available classes always come from `CLASS_NAMES` in `settings.py`. When using `--prefill`, the model's detections are filtered to only keep classes matching `CLASS_NAMES` (other detections are ignored).

```bash
python -m trocr_handwritten.parse.annotate data/raw/images/
```

To pre-fill bounding boxes using an existing model:

```bash
python -m trocr_handwritten.parse.annotate data/raw/images/ \
    --prefill --model models/20241119_v2_yolov10_50_finetuned.pt
```

| Shortcut | Action |
|----------|--------|
| 1-8 | Select class |
| Click + drag | Draw bbox |
| Click bbox | Select (move / resize corners) |
| Delete | Remove selected bbox |
| S | Save |
| Left / Right | Navigate images |
| Scroll | Zoom |

### 🚅 Training a layout parser

Training reads the `annotations.json` files from each split, creates YOLO-format images + labels, and fine-tunes a YOLO11 model.

Download a base model first (e.g. `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`):

```python
from ultralytics import YOLO
YOLO("yolo11n.pt")
```

Then train:

```bash
python -m trocr_handwritten.parse.train \
    --path-data data/layout \
    --images-dir data/raw/images \
    --model-base models/yolo11n.pt \
    --epochs 50 \
    --batch 8 \
    --device auto
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--path-data` | `data/layout` | Root directory with split annotations |
| `--images-dir` | `data/raw/images` | Source directory with original images |
| `--model-base` | `yolo11n.pt` | Base YOLO11 checkpoint (n/s/m/l/x) |
| `--epochs` | 50 | Number of training epochs |
| `--batch` | 8 | Batch size |
| `--device` | `auto` | Device (auto, cpu, cuda, mps) |
| `--patience` | 20 | Early stopping patience |
| `--freeze` | 0 | Freeze first N layers |

The best checkpoint is saved to `data/layout/models/best.pt`.

#### Evaluation

```bash
python -m trocr_handwritten.parse.metrics \
    --path-data data/layout \
    --model data/layout/models/best.pt \
    --split test
```

Outputs per-class Precision / Recall / F1 and overall metrics.

#### Push model to HuggingFace

```bash
python -m trocr_handwritten.parse.push_model \
    --model-path data/layout/models/best.pt \
    --repo-id agomberto/historical-layout-ft \
    --filename best_yolo11.pt
```

The model can then be loaded for inference by setting `hf_filename` in `LayoutParserSettings`.

#### Using a cloud GPU

```bash
ssh ubuntu@<instance-ip>
git clone https://github.com/handwrittenOCR/trocr_handwritten.git
cd trocr_handwritten
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python -m trocr_handwritten.parse.train \
    --images-dir data/raw/images --device auto --batch 16
```

Recommended batch sizes: A10 (24GB) → 8, A100 (80GB) → 16.

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

## 🤖 LLM-based OCR

### 🎯 Objective

The LLM OCR module provides an alternative approach to handwritten text recognition using vision-capable Large Language Models. Instead of using TrOCR, this module sends document images directly to LLM APIs (OpenAI, Google Gemini, or Mistral) for transcription.

### 🛠️ Supported Providers

| Provider | Default Model | Vision Support |
|----------|--------------|----------------|
| OpenAI | gpt-5.2 | Yes |
| Gemini | gemini-3-pro-preview | Yes |
| Mistral | mistral-large-latest | Yes |

### 📜 How to Use

1. **Set up API keys** in your `.env` file:

```bash
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
MISTRAL_API_KEY=your_mistral_key
```

2. **Run the OCR script**:

```bash
python -m trocr_handwritten.llm.ocr
```

#### Command-line Options

```bash
python -m trocr_handwritten.llm.ocr \
    --provider gemini \
    --model gemini-3-pro-preview \
    --input_dir data/processed/images \
    --pattern "*/*/*.jpg" \
    -n 10 \
    --max_concurrent 5
```

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | LLM provider (openai, gemini, mistral) | gemini |
| `--model` | Model name | Provider default |
| `--input_dir` | Root directory with images | data/processed/images |
| `--pattern` | Glob pattern for images | `*/*/*.jpg` |
| `-n` | Limit number of images | None (all) |
| `--max_concurrent` | Parallel API calls | 5 |
| `--prompt_path` | Custom prompt file | config/ocr.prompt |

### 📁 Output Structure

For each image `000.jpg`, a corresponding `000.md` file is created:

```bash
data/processed/images/
└── document_folder/
    └── Plein Texte/
        ├── 000.jpg
        ├── 000.md  # Transcription
        ├── 001.jpg
        └── 001.md
```

### 💰 Cost Tracking

The script automatically tracks API usage and displays costs at the end:

```
========================================
Cost Summary
========================================
Model: gemini-3-pro-preview
Total calls: 50
Input tokens: 125,000
Output tokens: 2,500
Estimated cost: $0.2800
```

### ⚠️ Error Handling

Failed transcriptions are logged to `data/processed/images/failed_ocr.json`:

```json
{
  "timestamp": "2025-01-19T16:50:00",
  "model": "gemini-3-pro-preview",
  "provider": "gemini",
  "failed_count": 2,
  "images": {
    "path/to/image.jpg": "error message"
  }
}
```

### 🔧 Custom Prompts

Edit `config/ocr.prompt` to customize the transcription instructions:

```
Tu es un expert en transcription de documents manuscrits historiques français.
Analyse l'image fournie et transcris fidèlement le texte manuscrit qu'elle contient.
...
```

### 📓 Usage in Notebook

To test LLM OCR on a single image in a Jupyter notebook:

```python
from pathlib import Path
from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.llm.factory import get_provider

settings = LLMSettings(provider="gemini", model_name="gemini-2.0-flash")
provider = get_provider(settings)

prompt = Path("config/ocr.prompt").read_text()
image_path = Path("data/processed/images/document/Plein Texte/000.jpg")

text, input_tokens, output_tokens = provider.ocr_image(image_path, prompt)
print(text)
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
