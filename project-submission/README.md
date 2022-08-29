Intro to Deep Learning | Project E2

# Facial Landmark Detection

### Submission

The submission includes two trained model files which can be tested with the demo notebook or using `torch.load(<path-to-model-file>)`.

The project code is split into 3 files with`data_loading.py` containing all functionality used for data-loading and preprocessing and `visualisation.py` the one concerning the visualisation of training and test data. `train.py` is a training script which can be executed from the command line and automatically fetches the dataset from [GitHub](https://github.com/ko-redtruck/facial-landmark-detection/raw/main/facial-keypoints-detection.zip).

### Execution

The usage of the training script is `python <path-to-training-script> [--reduce-data] [output-model-path]`.

Execution starts a new training run, saving the resulting model to `output-model-path` if supplied, with the default being `model.pickle`.

Additionally, the size of the dataset can be reduced from 2096 to 200 data points by providing the `-- reduce-data`-option when executing the script which is useful for testing on lightweight machines.
In the same spirit the current run configuration is tailored to be very resource friendly and not what we used for our training (example configurations below). The configuration can be changed by adjusting the desired values in the `config`-dictionary inside the python file.

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