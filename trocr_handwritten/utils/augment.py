## ADD TRAIN WITH LOWER CONSTRAST

import albumentations as A
import cv2
from tqdm import tqdm
from os.path import join
import pandas as pd

file_names, texts = [], []


def lower_contrast(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Declare an augmentation pipeline
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=[-0.5, -0.25], p=1),
        ]
    )

    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


def augment_train(PATH_DATA, train_df):
    for file_name, text in tqdm(
        train_df.loc[:, ["file_name", "text"]].values, total=len(train_df)
    ):
        image = cv2.imread(join(PATH_DATA, f"{file_name}"))
        image = lower_contrast(image)

        file_name = file_name.split("/")
        file_name = "/".join(file_name[:-1] + [f"lower_contrast_{file_name[-1]}"])

        # cv2.imwrite(join(PATH_DATA, f"{file_name}"), transformed_image)

        file_names.append(f"{file_name}")
        texts.append(text)

    augmented_train = pd.DataFrame([file_names, texts], index=["file_name", "text"]).T
    train_df = pd.concat((train_df, augmented_train))

    return train_df
