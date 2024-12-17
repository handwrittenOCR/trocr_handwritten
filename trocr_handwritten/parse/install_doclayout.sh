#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting installation...${NC}"

# Download Miniconda
echo -e "${GREEN}Downloading Miniconda...${NC}"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install Miniconda
echo -e "${GREEN}Installing Miniconda...${NC}"
bash ~/miniconda.sh -b -p $HOME/miniconda

# Initialize conda
echo -e "${GREEN}Initializing conda...${NC}"
source $HOME/miniconda/bin/activate
conda init

# Clone the repository
git clone https://github.com/agombert/DocLayout-YOLO.git

# Navigate to the repository
cd DocLayout-YOLO

# Create and activate environment
echo -e "${GREEN}Creating doclayout_yolo environment...${NC}"
conda create -n doclayout_yolo python=3.10 -y

# Need to source bashrc to ensure conda commands work
source ~/.bashrc

# Activate environment
echo -e "${GREEN}Activating environment...${NC}"
conda activate doclayout_yolo

# Install DocLayout-YOLO
echo -e "${GREEN}Installing DocLayout-YOLO...${NC}"
pip install -e .

# Clean up
echo -e "${GREEN}Cleaning up installation files...${NC}"
rm ~/miniconda.sh

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${GREEN}To activate the environment, use: conda activate doclayout_yolo${NC}"
