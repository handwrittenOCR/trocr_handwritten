
#Install datasets
conda install -c conda-forge datasets -y

# Connect to HuggingFace hub
huggingface-cli login

# Prepare data for training
python trocr_handwritten/parse/prepare_data_for_training.py
