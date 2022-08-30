import sys, os
import torch
from data_loading import get_facial_landmark_detection_data
from data_visualisation import plot_images, predict_and_draw_facial_landmarks

USAGE_MSG = f"""usage: python {sys.argv[0]} model_file test_image1 [test_image2 ...]
       python {sys.argv[0]} model_file --use-dataset [data_dir]
\tif the --use-dataset option is supplied the first 3 pictures of the original dataset will be used and the predictions plotted against the actual labels\n"""

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit("Missing arguments...\n" + USAGE_MSG)
    
    model_file = sys.argv[1]
    if not os.path.isfile(model_file):
        sys.exit("Error: Model file does not exist.\n" + USAGE_MSG)
    
    model = torch.load(model_file)

    images, labels = None
    if sys.argv[2] != "--use-dataset":
        images = [Image.open(image_path) for image_path in sys.argv[2:]]
    else:
        images, labels = get_facial_landmark_detection_data(r"./data" if len(sys.argv) != 4 else sys.argv[3])
        images, labels = images[:3], labels[:3]

    predicted_images = predict_and_draw_facial_landmarks(model, *images)
    
    plot_images(predicted_images, labels)
