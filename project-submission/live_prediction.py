# import the opencv library
import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from data_visualisation import predict_and_draw_facial_landmarks

USAGE_MSG = f"python {sys.argv[0]} model_file"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict_image(image, net):
    resized_image = transforms.functional.center_crop(image, [224])
    return predict_and_draw_facial_landmarks(resized_image, net=net, device=device)[0]


def main(target_size: [], net=None):
    if net is None:
        if len(sys.argv) != 2:
            sys.exit("Unexpected number of arguments received.\n" + USAGE_MSG)

        model_file = sys.argv[1]
        if not os.path.isfile(model_file):
            sys.exit(f"Error: File {model_file} does not exist.\n" + USAGE_MSG)

        net = torch.load(model_file, map_location=device)

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
    main(target_size=[224 * 3])
