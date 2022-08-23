#!/bin/python3

import pandas as pd
import zipfile
import numpy as np
import wget, os

import torch
from torchvision.transforms import functional as tf, Compose, ColorJitter, RandomErasing, InterpolationMode
from numpy.random import randint, random
from typing import Tuple
import math

from data_visualisation import plot_images


# ---DATA-LOADING---

def extract_zip(file_path: str, target_dir: str):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)


def _convert_to_numpy(data: pd.DataFrame):
    X = [np.fromstring(image, dtype=np.uint8, sep=' ').reshape(96, 96) for image in data["Image"]]
    X = np.reshape(X, (-1, 96, 96))

    Y = np.delete(data.values, 30, axis=1)
    Y = Y.reshape(-1, 30).astype('float32')
    return X, Y


def load_data_and_filter_from_csv(path: str):
    data = pd.read_csv(path)
    data.dropna(inplace=True)
    return _convert_to_numpy(data)


# Download zip file from GitHub and extract training data from csv file
def get_facial_landmark_detection_data(data_dir: str):
    data_dir += '/' if data_dir[-1] != '/' else ''
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if not os.path.isfile(f"{data_dir}facial-keypoints-detection.zip"):
        url = r"https://github.com/ko-redtruck/facial-landmark-detection/raw/main/facial-keypoints-detection.zip"
        wget.download(url, data_dir)

    extract_zip(f"{data_dir}facial-keypoints-detection.zip", data_dir)
    return load_data_and_filter_from_csv(f"{data_dir}training.zip")


# ---PREPROCESSING---

# Basic augmentation steps for and image (no adjustment of labeling necessary)
def transform_image(image: np.ndarray):
    transform = Compose([
        ColorJitter(0.5, 0.5, 0.5, 0.5),
        RandomErasing(scale=(0.01, 0.02))
    ])

    image = tf.to_tensor(image)
    image = tf.adjust_sharpness(image, 2 * random())
    return transform(image)


# Scale an image to a random size from the original up to max_size
def random_scale_datapoint(image: torch.Tensor, labeling: np.ndarray, max_size: Tuple[int, int]):
    new_size_x = randint(image.shape[-1], max_size[1], dtype='uint8')
    scaling = new_size_x / image.shape[-1]

    new_size_y = round(image.shape[-2] * scaling)

    return tf.resize(image, [new_size_y, new_size_x]), labeling * scaling


# size_factor multiplies the dataset
def augment_data(images, labels, max_image_size: Tuple[int, int], size_multiplier: int = 1):
    augmented_images = []
    augmented_labels = []

    for _ in range(size_multiplier):
        for i in range(len(images)):
            image = transform_image(images[i])
            image, labeling = random_scale_datapoint(image, labels[i], max_image_size)

            augmented_images.append(image)
            augmented_labels.append(labeling)
    return augmented_images, augmented_labels


# The following functions are used to pad images to a desired size and adjust the corresponding labels
def random_pad_datapoint_from_edge(image: torch.Tensor, labeling: np.ndarray, new_size: Tuple[int, int],
                                   offset: Tuple[int, int]):
    padding_right = new_size[-1] - image.shape[-1] - offset[-1]
    padding_bottom = new_size[-2] - image.shape[-2] - offset[-2]

    padded_image = tf.pad(image, [offset[-1], offset[-2], padding_right, padding_bottom], padding_mode="edge")
    adjusted_labeling = np.zeros(labeling.shape, dtype=labeling.dtype)
    adjusted_labeling[:-1:2] = labeling[:-1:2] + offset[-1]
    adjusted_labeling[1::2] = labeling[1::2] + offset[-2]

    return padded_image, adjusted_labeling


def random_pad_datapoint_with_noise(image: torch.Tensor, labeling: np.ndarray, new_size: Tuple[int, int],
                                    offset: Tuple[int, int]):
    def _random_pad_image_with_noise(image: torch.Tensor, new_size: Tuple[int, int], offset_x: int = 0,
                                     offset_y: int = 0):
        random_image = tf.to_tensor(randint(256, dtype='uint8', size=new_size))

        random_image[:, offset_y:image.shape[-2] + offset_y, offset_x:image.shape[-1] + offset_x] = image
        return random_image

    padded_image = _random_pad_image_with_noise(image, new_size, offset_x=offset[-1], offset_y=offset[-2])
    adjusted_labeling = np.zeros(labeling.shape, dtype=labeling.dtype)
    adjusted_labeling[:-1:2] = labeling[:-1:2] + offset[-1]
    adjusted_labeling[1::2] = labeling[1::2] + offset[-2]

    return padded_image, adjusted_labeling


# 3/4 of the images are padded with torchvision "edge"-mode and 1/4 with random integer values for every pixel
def random_pad_datapoint(image: torch.Tensor, labeling: np.ndarray, new_size: Tuple[int, int]):
    def _random_offset(image_dimensions, new_size: Tuple[int, int]):
        min_offset = (0, 0)
        max_offset = np.subtract(new_size, image_dimensions)

        offset_y = randint(min_offset[0], max_offset[0])
        offset_x = randint(min_offset[1], max_offset[1])
        return offset_y, offset_x

    if randint(4) == 3:
        return random_pad_datapoint_with_noise(image, labeling, new_size, _random_offset(image.shape[-2:], new_size))
    return random_pad_datapoint_from_edge(image, labeling, new_size, _random_offset(image.shape[-2:], new_size))


# extra_padding is used to crop some images at the edge
def random_pad_data(images, labels, new_size: Tuple[int, int], extra_padding: int = 0):
    padded_images = []
    adjusted_labels = []

    for i in range(len(images)):
        image, labeling = random_pad_datapoint(images[i], labels[i],
                                               (new_size[0] + 2 * extra_padding, new_size[1] + 2 * extra_padding))
        image = tf.center_crop(image, list(new_size))
        labeling = labeling - extra_padding

        padded_images.append(image)
        adjusted_labels.append(labeling)
    return torch.stack(padded_images), adjusted_labels


# The following functions are used to rotate images and the corresponding labels
def _rotate_point(point, angle):
    ox = oy = 112
    px, py = point
    angle_radians = math.radians(angle)

    qx = ox + math.cos(angle_radians) * (px - ox) - math.sin(angle_radians) * (py - oy)
    qy = oy + math.sin(angle_radians) * (px - ox) + math.cos(angle_radians) * (py - oy)
    return qx, qy


def _rotate_labeling(labeling, angle):
    rotated_labeling = []
    points = zip(labeling[:-1:2], labeling[1::2])

    for point in points:
        rotated_x, rotated_y = _rotate_point(point, angle)
        rotated_labeling.append(rotated_x)
        rotated_labeling.append(rotated_y)
    return np.array(rotated_labeling)


def rotate_data(images, labels):
    rotated_images = []
    rotated_labels = []

    for i in range(len(images)):
        lable_angle = randint(360)
        image_angle = 360 - lable_angle

        rotated_images.append(
            tf.rotate(images[i], image_angle, fill=random(), interpolation=InterpolationMode.BILINEAR))
        rotated_labels.append(_rotate_labeling(labels[i], lable_angle))
    return torch.stack(rotated_images), rotated_labels


# ---FINAL OUTPUT---

# Transform single-color-channel images to RGB using tensor magic ;)
def _transform_single_channel_to_rgb(images: torch.Tensor):
    transformed_images = torch.zeros((len(images), 3, 224, 224))
    transformed_images[:, :] = images[:]
    return transformed_images


def preprocess_data(images: np.ndarray, labels: np.ndarray, target_size: Tuple[int, int], dataset_multiplier: int = 1,
                    max_crop_length: int = 0, display_samples: bool = True):
    images, labels = augment_data(images, labels, target_size, dataset_multiplier)
    images, labels = random_pad_data(images, labels, target_size, max_crop_length)
    images, labels = rotate_data(images, labels)
    images = _transform_single_channel_to_rgb(images)

    if display_samples:
        plot_images(images, labels, 3)
    print("Final image shape:", images[0].shape)
    return images, labels
