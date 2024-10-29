---
license: mit
datasets:
- agomberto/FrenchCensus-handwritten-texts
language:
- fr
pipeline_tag: image-to-text
tags:
- pytorch
- transformers
- trocr
widget:
- src: >-
    https://raw.githubusercontent.com/agombert/trocr-base-printed-fr/main/sample_imgs/4.png
  example_title: Example 1
- src: >-
    https://raw.githubusercontent.com/agombert/trocr-base-printed-fr/main/sample_imgs/5.jpg
  example_title: Example 2
metrics:
- cer
- wer
---

# TrOCR base handwritten for French

## Overview

TrOCR handwritten has not yet released for French, so we trained a French model for PoC purpose. Based on this model, it is recommended to collect more data to additionally train the 1st stage or perform fine-tuning as the 2nd stage.

It's a special case of the [English handwritten trOCR model](https://huggingface.co/microsoft/trocr-base-handwritten) introduced in the paper [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) by Li et al. and first released in [this repository](https://github.com/microsoft/unilm/tree/master/trocr) as a TrOCR model fine-tuned on the [IAM dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

We decided to fine-tuned on two datasets:
1. [French Census dataset](https://zenodo.org/record/6581158) from Constum et al. We created a [dataset on the hub](https://huggingface.co/datasets/agomberto/FrenchCensus-handwritten-texts) too.
2. A dataset soon to come on French archives

## Model description

The TrOCR model is an encoder-decoder model, consisting of an image Transformer as encoder, and a text Transformer as decoder. The image encoder was initialized from the weights of BEiT, while the text decoder was initialized from the weights of RoBERTa.

Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder. Next, the Transformer text decoder autoregressively generates tokens.

## Intended uses & limitations

You can use the raw model for optical character recognition (OCR) on single text-line images.

## Parameters
We used heuristic parameters without separate hyperparameter tuning.
- learning_rate = 4e-5
- epochs = 10
- fp16 = True
- max_length = 32
- split train/dev: 90/10

## Metrics

For the dev set we got those results
- size of the test set: 1470 examples
- CER: 0.09
- WER: 0.23

For the test set (from French Census only) we got those results
- size of the test set: 730 examples
- CER: 0.12
- WER: 0.26

### How to use

Here is how to use this model in PyTorch:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
from PIL import Image
import requests

url = "https://github.com/agombert/trocr-base-printed-fr/blob/main/sample_imgs/5.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('agomberto/trocr-base-handwritten-fr')
tokenizer = AutoTokenizer.from_pretrained('agomberto/trocr-base-handwritten-fr')

pixel_values = (processor(images=image, return_tensors="pt").pixel_values)
generated_ids = model.generate(pixel_values)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```