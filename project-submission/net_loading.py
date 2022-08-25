from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
import torch
import torch.nn as nn

def add_final_layer(network, layer):
    network.fc = layer
    return network


def load_net(net: str, net_artifact: str):
    # Define parameter mappings
    final_layer = "Lin-ReLu-Lin": nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 30)
        )
  
    networks = {
        "ResNet18": add_final_layer(resnet18(weights=ResNet18_Weights.DEFAULT), final_layer),
        "ResNet34": add_final_layer(resnet34(weights=ResNet34_Weights.DEFAULT), final_layer)
    }

    network = networks[net].to(device)
    state_dict = torch.load(net_artifact, map_location=device)
    network.load_state_dict(state_dict)

    return networks[net].to(device)
