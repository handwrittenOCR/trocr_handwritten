#inference.py

from doclayout_yolo import YOLOv10
model = YOLOv10("PATH/TO/20241105_yolov10_finetuned.pt")

det_res = model.predict(
    "PATH_TO_IMAGE_FOLDER",   # Image to predict
    imgsz=1024,        # Prediction image size
    conf=0.2,          # Confidence threshold
    device="cpu"»    # Device to use (e.g., 'cuda:0' or 'cpu')
)

from matplotlib import pyplot as plt

# Annotate the result
annotated_frames = []
for res in det_res:
    annotated_frame = res.plot(pil=True, line_width=5, font_size=20)
    plt.figure(figsize=(15, 15))
    plt.imshow(annotated_frame)
    plt.axis('off')
    plt.title("Image avec polygons")
    plt.show()
