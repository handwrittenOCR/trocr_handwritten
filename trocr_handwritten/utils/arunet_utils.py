# added on 23/10/2024 because encontering issues with torchvision.transforms.functional

from os.path import join
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import cv2
import random
import numpy as np
import PIL as PIL
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 23/10/2024: added shapely to avoid issues with LineString

from shapely.geometry import LineString

from trocr_handwritten.utils.arunet import ARUNET, RUNET, UNET
from trocr_handwritten.utils.xmlPage import pageData
import trocr_handwritten.utils.polyapprox as pa


def create_aru_net(in_channels=3, out_channels=1, model_kwargs={}):
    model = model_kwargs.get("model", "aru")

    scale_space_num = model_kwargs.get("scale_space_num", 6)
    res_depth = model_kwargs.get("res_depth", 3)
    featRoot = model_kwargs.get("featRoot", 8)
    filter_size = model_kwargs.get("filter_size", 3)
    pool_size = model_kwargs.get("pool_size", 2)
    activation_name = model_kwargs.get("activation_name", "relu")
    activation = nn.ReLU()  # default is relu
    if activation_name == "elu":
        activation = nn.ELU()
    num_scales = model_kwargs.get("num_scales", 5)

    if "aru" in model:
        print("Using ARU-Net")
        return ARUNET(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_space_num=scale_space_num,
            res_depth=res_depth,
            featRoot=featRoot,
            filter_size=filter_size,
            pool_size=pool_size,
            activation=activation,
            num_scales=num_scales,
        )
    elif "ru" in model:
        print("Using RU-Net")
        return RUNET(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_space_num=scale_space_num,
            res_depth=res_depth,
            featRoot=featRoot,
            filter_size=filter_size,
            pool_size=pool_size,
            activation=activation,
            final_conv_bool=True,
        )
    else:
        print("Using U-Net")
        return UNET(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_space_num=scale_space_num,
            res_depth=res_depth,
            featRoot=featRoot,
            filter_size=filter_size,
            pool_size=pool_size,
            activation=activation,
        )


class TestDataset(Dataset):
    def __init__(self, image_dir, image_height, image_width, padding):
        self.image_dir = image_dir
        self.image_height = image_height
        self.image_width = image_width
        self.padding = padding
        self.images = [x for x in os.listdir(image_dir) if "jpg" in x.lower()]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = join(self.image_dir, self.images[index])
        # RGB
        # image = np.array(Image.open(img_path).convert("RGB"))
        # Greyscale
        image = Image.open(img_path).convert("L")
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        image = transform(image)
        image = np.array(image, dtype=np.float32)

        if self.padding:
            # downscale
            max_size = max(self.image_height, self.image_width)
            resize = A.LongestMaxSize(max_size=max_size, p=1)
            downscaled_version = resize(image=image)
            image = downscaled_version["image"]

            # normalization
            norm = A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0)
            downscaled_version = norm(image=image)
            image = downscaled_version["image"]

            # padding
            maxH = self.image_height
            maxW = self.image_width
            padH = maxH - image.shape[0]
            padW = maxW - image.shape[1]
            if padH + padW > 0:
                if padH < 0:
                    padH = 0
                if padW < 0:
                    padW = 0
                npad = ((0, padH), (0, padW))
                image = np.pad(image, npad, mode="constant", constant_values=1)

            augmentation = A.Compose([ToTensorV2()])
            augmentations = augmentation(image=image)
            image = augmentations["image"]

        else:
            test_transform = A.Compose(
                [
                    A.Resize(height=self.image_height, width=self.image_width),
                    # A.Normalize(mean = [0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0), # getting a value between 0 and 1
                    A.Normalize(
                        mean=[0.0], std=[1.0], max_pixel_value=255.0
                    ),  # getting a value between 0 and 1
                    # automatically converts correct image format from PIL.Image (NHWC) to PyTorch format (NCHW)
                    ToTensorV2(),
                ]
            )

            augmentations = test_transform(image=image)
            image = augmentations["image"]

        return image


def get_test_loaders(
    test_dir,
    image_height,
    image_width,
    padding,
    num_workers=64,
    pin_memory=True,
):
    test_ds = TestDataset(
        image_dir=test_dir,
        image_height=image_height,
        image_width=image_width,
        padding=padding,
    )

    test_loader = DataLoader(
        test_ds, num_workers=num_workers, pin_memory=pin_memory, shuffle=False
    )

    return test_loader


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def save_test_predictions_as_imgs(
    loader, model, image_height, image_width, padding, output_dir, device="cuda"
):
    """Saves images predicted by the testing model. If the model has been trained with random downsampling, padding must be enabled."""
    model.eval()
    loop = tqdm(loader)
    for idx, (x) in tqdm(enumerate(loop)):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        image_name = loader.dataset.images[idx]  # name of original image

        # get size of original image
        image_path = join(loader.dataset.image_dir, image_name)
        original_image = PIL.Image.open(image_path)
        width, height = original_image.size

        # delete data endings
        if (
            str.endswith(image_name, ".jpg")
            or str.endswith(image_name, ".JPG")
            or str.endswith(image_name, ".png")
        ):
            image_name = image_name[: len(image_name) - 4]
        elif str.endswith(image_name, ".jpeg"):
            image_name = image_name[: len(image_name) - 5]

        if padding:
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                ]
            )

            preds = transform(preds[0])

            preds = np.array(preds)

            # downscale like in train to calculate padding and unpad afterwards
            max_size = max(image_height, image_width)
            resize = A.LongestMaxSize(max_size=max_size, p=1)
            downscaled_version = resize(image=np.array(original_image))
            downscaled_image = downscaled_version["image"]

            # calculated used padding
            maxH = image_height
            maxW = image_width
            padH = maxH - downscaled_image.shape[0]
            padW = maxW - downscaled_image.shape[1]

            # now unpad the created image
            preds = np.array(preds)
            preds = preds[0 : image_height - padH, 0 : image_width - padW]
            # preds = transforms.functional.crop(preds, top = 0, left = 0, height = padH, width=padW),
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=(height, width)),
                    transforms.ToTensor(),
                ]
            )

            preds = transform(preds)

        else:
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=(height, width)),
                    transforms.ToTensor(),
                ]
            )

            preds = transform(preds[0])

        # generate xml
        gen_page(
            in_img_path=image_path,
            line_mask=preds[0, :, :],
            id=image_name,
            output_dir=output_dir,
        )

    model.train()


def gen_page(in_img_path, line_mask, id, output_dir):
    """Code from: https://github.com/imagine5am/ARU-Net"""
    in_img = cv2.imread(in_img_path)
    (in_img_rows, in_img_cols, _) = in_img.shape
    # print('line_mask.shape:', line_mask.shape)

    cScale = np.array(
        [in_img_cols / line_mask.shape[1], in_img_rows / line_mask.shape[0]]
    )
    id = str(id)
    page = pageData(os.path.join(output_dir, id + ".xml"), creator="ARU-Net PyTorch")
    page.new_page(os.path.basename(in_img_path), str(in_img_rows), str(in_img_cols))

    kernel = np.ones((5, 5), np.uint8)
    validValues = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    # lines = line_mask.copy()
    lines = torch.clone(line_mask)
    lines[line_mask > 0.1] = 1
    # lines = lines.astype(np.uint8)
    lines = lines.to(torch.uint8)
    lines = lines.cpu().numpy()

    # plt.axis("off")
    # plt.imshow(lines, cmap='gray')
    # plt.show()

    r_id = 0
    lin_mask = np.zeros(line_mask.shape, dtype="uint8")

    reg_mask = np.ones(line_mask.shape, dtype="uint8")
    res_ = cv2.findContours(
        np.uint8(reg_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(res_) == 2:
        contours, hierarchy = res_
    else:
        _, contours, hierarchy = res_

    for cnt in contours:
        min_area = 0.01
        # --- remove small objects
        if cnt.shape[0] < 4:
            continue
        if cv2.contourArea(cnt) < min_area * line_mask.shape[0]:
            continue

        cv2.minAreaRect(cnt)
        # --- soft a bit the region to prevent spikes
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # box = np.array((rect[0][0], rect[0][1], rect[1][0], rect[1][1])).astype(int)
        r_id = r_id + 1
        approx = (approx * cScale).astype("int32")
        reg_coords = ""
        for x in approx.reshape(-1, 2):
            reg_coords = reg_coords + " {},{}".format(x[0], x[1])

        cv2.fillConvexPoly(lin_mask, points=cnt, color=(1, 1, 1))
        lin_mask = cv2.erode(lin_mask, kernel, iterations=1)
        lin_mask = cv2.dilate(lin_mask, kernel, iterations=1)
        reg_lines = lines * lin_mask

        resl_ = cv2.findContours(reg_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(resl_) == 2:
            l_cont, l_hier = resl_
        else:
            _, l_cont, l_hier = resl_

        # IMPORTANT l_cont, l_hier
        if len(l_cont) == 0:
            continue

        # --- Add region to XML only is there is some line
        uuid = "".join(random.choice(validValues) for _ in range(4))
        text_reg = page.add_element(
            "TextRegion", "r" + uuid + "_" + str(r_id), "full_page", reg_coords.strip()
        )
        n_lines = 0
        for l_id, l_cnt in enumerate(l_cont):
            if l_cnt.shape[0] < 4:
                continue
            if cv2.contourArea(l_cnt) < 0.01 * line_mask.shape[0]:
                continue
            # --- convert to convexHull if poly is not convex
            if not cv2.isContourConvex(l_cnt):
                l_cnt = cv2.convexHull(l_cnt)
            lin_coords = ""
            l_cnt = (l_cnt * cScale).astype("int32")
            # IMPORTANT
            (is_line, approx_lin) = get_baseline(in_img, l_cnt)

            if is_line is False:
                continue

            is_line, l_cnt = build_baseline_offset(approx_lin, offset=50)
            if is_line is False:
                continue
            for l_x in l_cnt.reshape(-1, 2):
                lin_coords = lin_coords + " {},{}".format(l_x[0], l_x[1])
            uuid = "".join(random.choice(validValues) for _ in range(4))
            text_line = page.add_element(
                "TextLine",
                "l" + uuid + "_" + str(l_id),
                "full_page",
                lin_coords.strip(),
                parent=text_reg,
            )
            # IMPORTANT
            baseline = pa.points_to_str(approx_lin)
            page.add_baseline(baseline, text_line)
            n_lines += 1
    page.save_xml()


def get_baseline(Oimg, Lpoly):
    """Code from: https://github.com/imagine5am/ARU-Net"""
    # --- Oimg = image to find the line
    # --- Lpoly polygon where the line is expected to be
    minX = Lpoly[:, :, 0].min()
    maxX = Lpoly[:, :, 0].max()
    minY = Lpoly[:, :, 1].min()
    maxY = Lpoly[:, :, 1].max()
    mask = np.zeros(Oimg.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, Lpoly, (255, 255, 255))
    cv2.bitwise_and(Oimg, mask)
    bRes = Oimg[minY:maxY, minX:maxX]
    bMsk = mask[minY:maxY, minX:maxX]
    bRes = cv2.cvtColor(bRes, cv2.COLOR_RGB2GRAY)
    _, bImg = cv2.threshold(bRes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, cols = bImg.shape
    # --- remove black halo around the image
    bImg[bMsk[:, :, 0] == 0] = 255
    Cs = np.cumsum(abs(bImg - 255), axis=0)
    maxPoints = np.argmax(Cs, axis=0)
    np.zeros(bImg.shape)
    points = np.zeros((cols, 2), dtype="int")
    # --- gen a 2D list of points
    for i, j in enumerate(maxPoints):
        points[i, :] = [i, j]
    # --- remove points at post 0, those are very probable to be blank columns
    points2D = points[points[:, 1] > 0]
    if points2D.size <= 15:
        # --- there is no real line
        return (False, [[0, 0]])

    # --- take only 100 points to build the baseline
    max_vertex = 30
    num_segments = 4
    if points2D.shape[0] > max_vertex:
        points2D = points2D[
            np.linspace(0, points2D.shape[0] - 1, max_vertex, dtype=np.int32)
        ]
    (approxError, approxLin) = pa.poly_approx(points2D, num_segments, pa.one_axis_delta)

    approxLin[:, 0] = approxLin[:, 0] + minX
    approxLin[:, 1] = approxLin[:, 1] + minY
    return (True, approxLin)


def build_baseline_offset(baseline, offset=50):
    """
    build a simple polygon of width $offset around the
    provided baseline, 75% over the baseline and 25% below.
    Code from: https://github.com/imagine5am/ARU-Net
    """
    try:
        line = LineString(baseline)
        up_offset = line.parallel_offset(offset * 0.75, "right", join_style=2)
        bot_offset = line.parallel_offset(offset * 0.25, "left", join_style=2)
    except Exception:
        # --- TODO: check if this baselines can be saved
        return False, None
    if (
        up_offset.type != "LineString"
        or up_offset.is_empty is True
        or bot_offset.type != "LineString"
        or bot_offset.is_empty is True
    ):
        return False, None
    else:
        up_offset = np.array(up_offset.coords).astype(np.int32)
        bot_offset = np.array(bot_offset.coords).astype(np.int32)
        return True, np.vstack((up_offset, bot_offset))


def combineImages(original_image_path, baseline_image_path, folder, image_name):
    """Combine result image with input image so baselines are drawn on the original image."""

    aImgPath = original_image_path

    # get the original image
    input_image = cv2.imread(aImgPath)

    # get the output image (baselines)
    output_image = cv2.imread(baseline_image_path)

    # values between 0 and 1
    image_mask = output_image / 255

    # get a red picture with the size of the output image
    color = np.zeros_like(output_image)
    color[:, :, 2] += 255

    combined_image = (1 - image_mask) * input_image + image_mask * color
    combined_image = np.uint8(combined_image)

    save_location = os.path.join(folder, image_name + "_combined.png")
    cv2.imwrite(save_location, combined_image)
