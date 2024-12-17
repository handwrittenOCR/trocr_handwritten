python prepare_data.py

DATA_PATH="data"

## model path
MODEL_PATH="models/doclayout_yolo_docstructbench_imgsz1024.pt"

## output path
OUTPUT_PATH="yolo_ft"

## Epochs
EPOCHS=50

## Patience
PATIENCE=5

## Batch size
BATCH_SIZE=8

python train.py --data $DATA_PATH/config --model m-doclayout --epoch $EPOCHS --image-size 1024 --batch-size $BATCH_SIZE --patience $PATIENCE --project $OUTPUT_PATH --optimizer Adam --lr0 0.001 --pretrain $MODEL_PATH --device 0

python push_model.py --model-path yolo_ft/best.pt --repo-id agomberto/historical-layout
