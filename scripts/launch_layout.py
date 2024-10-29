#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import shutil
import subprocess
import torch
from PIL import Image
import pandas as pd
from os.path import join, dirname
from tqdm import tqdm




### TODO: 
# Paralléliser tout ça 
# multiprocessing multithreading (dodo pense qu'il faut threader)
# asynchrone (dodo pense qu'il faut asynchrone)
# loader les pages en batch (utiliser les subfolders comme batch)


# Set up paths
PATH_REPO = "/work/m24063/m24063bglm/trocr_handwritten"
PATH_MODELS = join(PATH_REPO, 'models')
PATH_UTILS = join(PATH_REPO, 'trocr_handwritten', 'utils')
PATH_PARSE = join(PATH_REPO, 'trocr_handwritten', 'parse')
PATH_OCRIZING = join(PATH_REPO, 'trocr_handwritten', 'trocr')
PATH_DATA = join(PATH_REPO, 'data/mar_hopitaux_deces')
PATH_PAGES = join(PATH_DATA, 'PAGES')
PATH_XML = join(PATH_DATA, 'XML')
PATH_LINES = join(PATH_DATA, 'lines')
PATH_OCR_OUTPUT = join(PATH_DATA, 'OCRized')

sys.path.append(PATH_UTILS)
sys.path.append(PATH_PARSE)

# Load configuration
with open(join(PATH_PARSE, "config.json")) as f:
    config = json.load(f)

if torch.cuda.is_available():
    config['DEVICE'] = "cuda"
else:
    config['DEVICE'] = "cpu"

model_kwargs = dict(
    scale_space_num=config.get("SCALE_SPACE_NUM", 6),
    res_depth=config.get("RES_DEPTH", 3),
    feat_root=config.get("FEAT_ROOT", 8),
    filter_size=config.get("FILTER_SIZE", 3),
    pool_size=config.get("POOL_SIZE", 2),
    activation_name=config.get("ACTIVATION_NAME", "relu"),
    model=config.get("MODEL", "aru"),
    num_scales=config.get("NUM_SCALES", 5),
)

# Import required modules
from trocr_handwritten.utils.arunet_utils import (
    create_aru_net,
    get_test_loaders,
    load_checkpoint,
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
from trocr_handwritten.parse import parse_page

# Initialize model
device = torch.device(config['DEVICE'])
model = create_aru_net(in_channels=1, out_channels=1, model_kwargs=model_kwargs).to(device)
checkpoint = torch.load(join(PATH_MODELS, "cbad_2019.tar"), map_location=device)
load_checkpoint(checkpoint, model)

# Process folders for parsing



print("Processing completed.")