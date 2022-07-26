Intro to Deep Learning | Project E2

# Facial Landmark Detection

### Submission

The submission includes two trained model files (downloadable via the link in the model.txt file) which can be tested with the submitted demo notebook, the livestream code in `live_prediction.py` or using the testing script (`test.py`) which is part of the project code. There are also test images provided in the `testImages` directory.

The project code is split into 6 files with `data_loading.py` containing all functionality used for data-loading and preprocessing, `visualisation.py` the one concerning the visualisation and prediction of training and test data and `video_processing.py` having functions used to extraxt frames from videos and reconstruct them. `train.py` is a training script which can be executed from the command line and automatically fetches the dataset from [GitHub](https://github.com/ko-redtruck/facial-landmark-detection/raw/main/facial-keypoints-detection.zip). The script `test.py` can be used to test a model file on images and `live_prediction.py` starts a video stream, where the live webcam feed is being predicted.

### Execution

The usage of the training script is `python training_script [--reduce-data] [output_model]`.

Execution starts a new training run, saving the resulting model to `output_model` if supplied, with the default being `model.pickle`.

Additionally, the size of the dataset can be reduced from 2096 to 200 data points by providing the `--reduce-data`-option when executing the script which is useful for testing on lightweight machines.
In the same spirit the current run configuration is tailored to be very resource friendly and not what we used for our training (example configurations below). The configuration can be changed by adjusting the desired values in the `config`-dictionary inside the python file.

The testing script can be used with any JPG or PNG images or sample pictures from the original dataset, which will be fetched automatically (if not present in the provided data directory).
```
usage: python testing_script model_file test_image1 [test_image2 ...] 
       python testing_script model_file --use-dataset [data_dir] 
    if the --use-dataset option is supplied the first 3 pictures of the original dataset will be used and the predictions plotted against the actual labels
```
(if `data_dir` is not provided with the `--use-dataset`-option `./data` will be used)

The `live_prediction.py` script also only requires a model file as an argument and will take some time, before a new window should open, where you will be able to watch the model perform its predicitons live on your webcam. We found it a little bit hard and tricky to setup all the dependencies and environment stuff for this script, so for troubleshooting purposes, we have provided the output of the `conda list` command from a windows machine, that this script worked with. We used python version is 3.9.12 and conda version 4.13.0

### Run Configuration

The hyper-parameters for training are explicitly saved in the dictionary and can be changed directly, while the the desired network, optimizer and learning-rate scheduler is selected by supplying the respective key from the corresponding dictionary (`networks`, `optimizers`, etc.).*

The configurations used to train the submitted models:
- ResNet18
```
config = {
        "NET": "ResNet18",
        "FC_LAYER": "Lin-ReLu-Lin",
        "DATASET_MULTIPLIER": 10,
        "OPTIMIZER": "AdamW",
        "LOSS": "L1",
        "LR_SCHEDULER": "Cyclic",
        "LR_SCHEDULER_MODE": "exp_range",
        "LR_CYCLIC_SCHEDULER_STEP_UP_SIZE": 2000,
        "EPOCHS": 800,
        "BATCH_SIZE": 625,
        "MAX_LR": 0.012,
        "BASE_LR": 0.0001,
        "GAMMA": 0.99995,
        "WEIGHT_DECAY": 0.01
    }
```
- ResNet34
```
config = {
        "NET": "ResNet34",
        "FC_LAYER": "Lin-ReLu-Lin",
        "DATASET_MULTIPLIER": 10,
        "OPTIMIZER": "AdamW",
        "LOSS": "L1",
        "LR_SCHEDULER": "Cyclic",
        "LR_SCHEDULER_MODE": "exp_range",
        "LR_CYCLIC_SCHEDULER_STEP_UP_SIZE": 2400,
        "EPOCHS": 800,
        "BATCH_SIZE": 420,
        "MAX_LR": 0.012,
        "BASE_LR": 0.0001,
        "GAMMA": 0.99995,
        "WEIGHT_DECAY": 0.01
    }
```

(Please note that these runs were executed with 45GB of RAM and a GPU with 16GB of memory)

\* Not all hyper-parameters are relevant, depending on the choice of LR-scheduler.
