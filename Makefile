all: yolo llm

yolo:
	python trocr_handwritten/parse/layout_parser.py

llm:
	python trocr_handwritten/llm/ocr.py

run: yolo llm
