#!/bin/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def plot_image(image: Image, labeling: np.ndarray = None):
    try:
        image = to_pil_image(image)
    except TypeError:
        pass
    plt.imshow(image, interpolation='nearest', cmap='gray')

    if labeling is not None:
        print(len(labeling))
        for i in range(0, len(labeling) - 1, 2):
            plt.plot(labeling[i + 0], labeling[i + 1], marker=".", color='cyan')
    plt.show()


def plot_images(images: np.ndarray, labels: np.ndarray = None, num: int = None):
    for i in range(num if num is not None else len(images)):
        plot_image(images[i], None if labels is None else labels[i])
