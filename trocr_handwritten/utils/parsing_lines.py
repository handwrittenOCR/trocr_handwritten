import numpy as np
import logging
import os
from os.path import join
from os import makedirs
from xml.dom import minidom
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from shutil import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def get_columns_coords(bboxes, nbins=150, th=5):
    """
    Objective: get the coords of the columns in the page if there is more than th occurences in the bin
    """

    coords = []

    xmaxs = np.array([x[2] for x in bboxes])
    xmins = np.array([x[0] for x in bboxes])

    # get histogram of the xmaxs and xmins (confounded)
    hist = np.histogram(np.concatenate((xmaxs, xmins)), bins=nbins)
    # get only the peaks - freq > th
    indexes = list(np.where(hist[0] > th)[0])

    # restrict density to one coord --> [3, 4, 5] --> [5] if in freq you have [6, 8, 12]
    diff = [j - i for i, j in zip(indexes[:-1], indexes[1:])]
    while 1 in diff:
        i = diff.index(1)
        indexes.pop(i)
        diff = [j - i for i, j in zip(indexes[:-1], indexes[1:])]

    # create intervals of columns [[x1, x2], [x2, x3], [x3, x4], ...]
    for _min, _max in zip(indexes[:-1], indexes[1:]):
        coords.append([hist[1][_min], hist[1][_max]])

    # check for a potential first column if there is a margin - spec to our pages
    if sum(hist[0][: indexes[0]]) >= th:
        coords = [[hist[1][0], hist[1][indexes[0]]]] + coords

    return coords, hist, indexes


def resize_columns(coords, hist, indexes, th=5, th_width=200):
    """
    Objective. if there is some overlap OR if there is a column shorter than th_width
    """

    # calculate columns width
    diffs = [x[1] - x[0] for x in coords]
    under_th_width = np.where(np.array(diffs) < th_width)[0][::-1]

    while len(under_th_width) > 0:
        for i in under_th_width:
            if i == 0 or i == len(coords):
                coords.pop(i)
            else:
                # if under threshold of width
                # just increase previous column to get two columns in 1
                coords[i - 1][1] = coords[i][1]
                coords.pop(i)
        diffs = [x[1] - x[0] for x in coords]
        under_th_width = np.where(np.array(diffs) < th_width)[0][::-1]

    if sum(hist[0][: indexes[0]]) >= th and coords[0][0] - hist[1][0] > th_width:
        coords = [[hist[1][0], coords[0][0]]] + coords

    return coords


def resize_bbox(bbox, coords):
    """
    Objective: find final bbox according to columns
    """
    x_min, x_max = bbox[0], bbox[2]
    diffs = np.array([min(x_max, x[1]) - max(x_min, x[0]) for x in coords])
    index = np.argsort(diffs)[::-1][0]
    bbox[0] = min(bbox[0], coords[index][0])
    bbox[2] = max(coords[index][1], bbox[2])

    return bbox, index


def bbox_split(bbox, coords, index):
    """
    Objective: if the line that goes through several columns split by column
    """

    index_min = np.argmin([abs(bbox[0] - coord[0]) for coord in coords])
    index_max = np.argmin([abs(bbox[2] - coord[1]) for coord in coords])
    bboxes = []

    for i in range(index_min, index_max + 1):
        x0 = min(bbox[0], coords[i][0]) if index_min == i else coords[i][0]
        x2 = max(bbox[2], coords[i][0]) if index_max == i else coords[i][1]
        _bbox = [x0, bbox[1], x2, bbox[3]]
        bboxes.append((_bbox, i))

    return bboxes


def is_overlap_picture(bbox1, bbox2, th=0.8):
    """
    Objective: quit overlapping pictures at more than a threshold
    """

    output = False

    overlap = [
        max(bbox1[0], bbox2[0]),
        max(bbox1[1], bbox2[1]),
        min(bbox1[2], bbox2[2]),
        min(bbox1[3], bbox2[3]),
    ]
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    areaoverlap = (overlap[2] - overlap[0]) * (overlap[3] - overlap[1])

    if areaoverlap / area1 > th and areaoverlap / area2 > th:
        output = True

    return output


def get_coords(data, with_text=True):
    """
    Objective: get the coordonates of each line from the parser
    """
    coords, texts = [], []

    page = data.getElementsByTagName("PcGts")[0].getElementsByTagName("Page")[0]
    regions = page.getElementsByTagName("TextRegion")
    if len(regions) == 0:
        regions = page.getElementsByTagName("TableRegion")
    if len(regions[0].getElementsByTagName("TableCell")) > 0:
        regions = regions[0].getElementsByTagName("TableCell")
    for region in regions:
        textLines = region.getElementsByTagName("TextLine")
        for textLine in textLines:
            coord = textLine.getElementsByTagName("Coords")[0].toprettyxml()[16:-4]
            coords.append(coord)
            if with_text:
                text = (
                    textLine.getElementsByTagName("TextEquiv")[-1]
                    .getElementsByTagName("Unicode")[0]
                    .toprettyxml()[9:-11]
                )
                texts.append(text)

    return coords, texts


def convert_coords(coords):
    """
    Objective: from all the coordinates, get the final bbox of the XML for a line
    """
    coords = np.array([[int(y) for y in x.split(",")] for x in coords.split(" ")])
    coords = [
        np.min(coords[:, 0]),
        np.min(coords[:, 1]),
        np.max(coords[:, 0]),
        np.max(coords[:, 1]),
    ]
    return coords


def increase_y(bboxes):
    """
    Objective: increase the height of the line by 1/3
    """
    ys = [bbox[3] - bbox[1] for bbox in bboxes]
    y_max = np.max(ys)
    ys = y_max - np.array(ys)
    # th = np.quantile(ys, 0.5)
    bboxes_updated = []
    for i, bbox in enumerate(bboxes):
        bbox[1], bbox[3] = bbox[1] - ys[i] / 3, bbox[3] + ys[i] / 3
        bboxes_updated.append(bbox)

    return bboxes_updated


def save_bbox_images(my_page, columns, image, path_lines):
    for column, _bboxes in columns.items():
        column_dir = join(path_lines, my_page, f"column_{column}")
        makedirs(column_dir, exist_ok=True)
        for i, bbox in enumerate(_bboxes):
            bbox_image = image.crop(bbox)
            bbox_image.save(join(column_dir, f"{my_page}_line_{i}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg"))

def display_image_with_bboxes(my_page, columns, path_pages):
    image_with_bboxes = Image.open(join(path_pages, f"{my_page}.jpg")).convert("RGB")
    draw = ImageDraw.Draw(image_with_bboxes, "RGB")

    colors = ["red", "blue", "yellow", "green", "orange"]

    for column, _bboxes in columns.items():
        for bbox in _bboxes:
            draw.rectangle(bbox, outline=colors[column], width=5)

    plt.imshow(image_with_bboxes)
    plt.show()

def handle_error(my_page, error, path_pages, path_lines):
    print(f"An error occurred with page {my_page}: {error}")
    makedirs(join(path_lines, "error"), exist_ok=True)
    copy(join(path_pages, f"{my_page}.jpg"), join(path_lines, "error"))
    
     
# 25/10/2024: added batch processing: 
# Handle subfolders in the PAGE, LINE, and XML folders by using os.walk().
# Preserve the folder structure in the output.
# Work with or without subfolders.
# Implement batch processing using ProcessPoolExecutor for parallel execution on the server.


    
