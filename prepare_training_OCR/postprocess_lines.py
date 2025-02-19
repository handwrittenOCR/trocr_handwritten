import json
from datasets import load_dataset
from PIL import Image

def crop_line_image(sample, crop_factor=0.1):
    """
    Crop the line image in the sample using adjusted bounding boxes.
    Returns the cropped image and its coordinates in the original image's coordinate system.
    """
    img = sample['image']
    img_width, img_height = img.size  # Get the size of the image

    cropped_image = img.crop((0,img_height*crop_factor/2, img_width, img_height*(1-crop_factor/2)))
    new_img_width, new_img_height = cropped_image.size
    
    #print(f"original_image: {img.size}")
    #print(f"cropped_image: {cropped_image.size}")
    #cropped_image.show()
    
    #translate the coordinates
    x1, y1, x2, y2 = sample['x1'], sample['y1'], sample['x2'], sample['y2']
    new_x1=x1
    new_y1=y1+img_height*crop_factor/2
    new_x2=x2
    new_y2=y2-img_height*crop_factor/2
    
    translated_coordinates=(new_x1, new_y1, new_x2, new_y2)
    
    #print(f"original_coordinates: {x1}, {y1}, {x2}, {y2}")
    #print(f"translated_coordinates: {translated_coordinates}")

    return cropped_image, translated_coordinates