#!/bin/python3


import torch, torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR
import sys, os

from data_loading import get_facial_landmark_detection_data, preprocess_data
from torch.utils.data import DataLoader, random_split

# Fetch, preprocess data and get dataloaders
def get_data_loaders(images, labels, batch_size, test_data_split=0.1, num_workers=2):
    dataset = list(zip(images, labels))
    training_data_size = int(len(dataset) * (1 - test_data_split))
    train, test = random_split(dataset, [training_data_size, len(dataset) - training_data_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader

if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        sys.exit("Wrong number of arguments provided...\nUsage: python <path to train.py> [--reduce-data] <output file>")

    if len(sys.argv) == 3 and sys.argv[1] != "--reduce-data":
        sys.exit(
            f"Received unexpected argument: {sys.argv[1]}\nUsage: python <path to train.py> [--reduce-data] <output file>")

    should_reduce_data = len(sys.argv) == 3

    model_file = sys.argv[-1]
    if os.path.isfile(model_file):
        sys.exit("Error: File already exists.\nAborting...")

    # Configuration of training run
    config = {
        "NET": "ResNet18",
        "FC_LAYER": "Lin-ReLu-Lin",
        "DATASET_MULTIPLIER": 1,
        "OPTIMIZER": "AdamW",
        "LOSS": "L1",
        "LR_SCHEDULER": "OneCycle",
        "LR_SCHEDULER_MODE": "exp_range",
        "LR_CYCLIC_SCHEDULER_STEP_UP_SIZE": 2400,
        "EPOCHS": 5,
        "BATCH_SIZE": 50,
        "MAX_LR": 0.007,
        "BASE_LR": 0.0001,
        "GAMMA": 0.99995,
        "WEIGHT_DECAY": 0.01
    }


    # Parameter mappings
    def add_fc(net, layer):
        net.fc = layer
        return net


    fc_layers = {
        "Lin-ReLu-Lin": nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 30)
        ),
        "Linear": nn.Linear(512, 30)
    }
    networks = {
        "ResNet18": add_fc(resnet18(pretrained=True), fc_layers[config["FC_LAYER"]]),
        "ResNet34": add_fc(resnet34(weights="DEFAULT"), fc_layers[config["FC_LAYER"]]),
        "ResNet50": add_fc(resnet50(weights="DEFAULT"), fc_layers[config["FC_LAYER"]])
    }
    optimizers = {
        "Adam": Adam(networks[config["NET"]].parameters(), weight_decay=config["WEIGHT_DECAY"]),
        "AdamW": AdamW(networks[config["NET"]].parameters(), weight_decay=config["WEIGHT_DECAY"])
    }
    loss_functions = {
        "MSE": nn.MSELoss(),
        "L1": nn.L1Loss()
    }

    # Set up training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = networks[config["NET"]].to(device)
    optimizer = optimizers[config["OPTIMIZER"]]
    loss_function = loss_functions[config["LOSS"]]
    num_workers = 0 if sys.platform.startswith('win') else 2


    


    images, labels = get_facial_landmark_detection_data("./data")
    print("Fetched Images, processing...")

    if should_reduce_data:
        images, labels = images[:200], labels[:200]
    preprocessed_images, preprocessed_labels = preprocess_data(images, labels, target_size=(224, 224),
                                                               dataset_multiplier=config["DATASET_MULTIPLIER"],
                                                               max_crop_length=30, display_samples=False)

    train_loader, test_loader = get_data_loaders(preprocessed_images, preprocessed_labels, batch_size=config["BATCH_SIZE"],
                                                 test_data_split=0.15, num_workers=num_workers)

    # Set up learning rate scheduler
    schedulers = {
        "OneCycle": OneCycleLR(optimizer, max_lr=config["MAX_LR"], steps_per_epoch=len(train_loader),
                               epochs=config["EPOCHS"]),
        "Cyclic": CyclicLR(optimizer, base_lr=config["BASE_LR"], max_lr=config["MAX_LR"],
                           step_size_up=config["LR_CYCLIC_SCHEDULER_STEP_UP_SIZE"], mode=config["LR_SCHEDULER_MODE"],
                           gamma=config["GAMMA"], cycle_momentum=False)
    }
    scheduler = schedulers[config["LR_SCHEDULER"]]


    # Training
    def compute_loss(inputs, labels, net, loss_function):
        outputs = net(inputs.to(device))
        labels = labels.to(device)

        return loss_function(outputs, labels)


    train_loss, test_loss = 0., 0.
    data_points = 0

    print("Training...")
    for epoch in range(config["EPOCHS"]):
        # train model
        train_loss = 0.
        net.train()
        for inputs, labels in train_loader:
            data_points += inputs.shape[0]
            loss = compute_loss(inputs, labels, net, loss_function)
            train_loss += loss

            # optimize model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # average loss
        train_loss /= len(train_loader)

        currentreduce_data_learning_rate = scheduler.get_last_lr()[0]

        # test model
        net.eval()
        with torch.no_grad():
            test_loss = 0.
            for inputs, labels in test_loader:
                test_loss += compute_loss(inputs, labels, net, loss_function)

            test_loss /= len(test_loader)

        # log error
        print(
            f'epoch: {epoch + 1}, train loss: {train_loss}, test loss: {test_loss}, learning rate: {current_learning_rate}')

    # Save model
    torch.save(net, model_file)
