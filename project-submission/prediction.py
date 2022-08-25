import torch
from PIL import ImageDraw
from torchvision import transforms

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
