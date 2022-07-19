import numpy as np # linear algebra
import pandas as pd
import zipfile

DATA_DIR = "./data"

with zipfile.ZipFile("facial-keypoints-detection.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

training_data = pd.read_csv(f'{DATA_DIR}/training.zip')

X = [np.fromstring(image, dtype=int, sep=' ').reshape(96,96,1) for image in training_data["Image"]]
X = np.reshape(X,(-1,96, 96, 1))

Y = training_data["left_eye_center_x"].values.reshape(-1,1)

#%%

#%%
##Hallo 
