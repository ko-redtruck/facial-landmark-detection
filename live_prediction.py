# import the opencv library
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
import cv2
import wandb
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def add_final_layer(network, layer):
    network.fc = layer
    return network


def get_net(net: str, version: str, final_layer: str):
    net_artifact_path = f"./artifacts/{net}-{version}/model"
    if not os.path.isfile(net_artifact_path):
        wandb.login()
        wandb.init()
        net_artifact = wandb.use_artifact(f"leo-team/facial-landmark-detection/{net}:{version}", type='model_state')
        net_artifact.download()

    # Define parameter mappings
    final_layers = {
        "Lin-ReLu-Lin": nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 30)
        ),
        "Linear": nn.Linear(512, 30)
    }
    networks = {
        "ResNet18": add_final_layer(resnet18(weights=ResNet18_Weights.DEFAULT), final_layers[final_layer]),
        "ResNet34": add_final_layer(resnet34(weights=ResNet34_Weights.DEFAULT), final_layers[final_layer])
    }

    network = networks[net].to(device)
    state_dict = torch.load(net_artifact_path, map_location=device)
    network.load_state_dict(state_dict)

    return networks[net].to(device)


def predict_facial_landmarks(*pil_images, net):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224)
    ])

    tensor_images = [preprocess(image.convert('L').convert('RGB')) for image in pil_images]
    image_batch = torch.stack(tensor_images, dim=0).to(device)
    net.eval()
    labels = net(image_batch)
    return labels.cpu().detach().numpy()


def add_labeling_to_image(image, labeling):
    draw = ImageDraw.Draw(image)
    points = zip(labeling[:-1:2], labeling[1::2])

    for point in points:
        x, y = point
        radius = 1
        upper_left_point = (x - radius, y - radius)
        lower_right_point = (x + radius, y + radius)
        draw.ellipse((upper_left_point, lower_right_point), fill=(255, 0, 0))

    return image


def predict_image(image, net):
    resized_image = transforms.functional.center_crop(image, [224])
    labeling = predict_facial_landmarks(resized_image, net=net)[0]
    labelled_image = add_labeling_to_image(resized_image, labeling)
    return labelled_image


def main(net, target_size: []):
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

    while True:
        ret, frame = vid.read()

        image = Image.fromarray(frame)
        predicted_image = predict_image(image, net)
        resized_image = transforms.functional.resize(predicted_image, target_size)

        cv2.imshow('Prediction', np.array(resized_image))

        if cv2.waitKey(1) & 0xFF == 27:
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(get_net("ResNet34", "v12", "Lin-ReLu-Lin"), target_size=[224 * 3])
