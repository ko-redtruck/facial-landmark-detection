#!/bin/python3

from PIL import Image, ImageDraw
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, center_crop


# Plot images and labels
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


# Add predictions to image
def predict_facial_landmarks(net, *pil_images):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224)
    ])

    tensor_images = [preprocess(image.convert('L').convert('RGB')) for image in pil_images]
    image_batch = torch.stack(tensor_images, dim=0).to(device)
    net.eval()
    labels = net(image_batch)
    return labels.cpu().detach().numpy()


def add_labeling_to_images(pil_images, labels):
    def get_ellipse_corners(center, radius):
        x, y = center
        upper_left_corner = (x - radius, y - radius)
        lower_right_corner = (x + radius, y + radius)

        return upper_left_corner, lower_right_corner
            
    labelled_images = []
    for image, labeling in zip(pil_images, labels):
        draw = ImageDraw.Draw(image)
        points = zip(labeling[:-1:2], labeling[1::2])

        for point in points:
            draw.ellipse(get_ellipse_corners(center, 1), fill=(255, 0, 0))
        labelled_images.append(image)
        
    return labelled_images


def predict_and_draw_facial_landmarks(net, center_crop_size, *pil_images):
    if center_crop_size != None:
        pil_images = [center_crop(image, [min(center_crop_size, min(image.size))]) for image in pil_images]

    labels = predict_facial_landmarks(net=net, *pil_images)
    labelled_images = add_labeling_to_images(pil_images, labels)
    return labelled_images
