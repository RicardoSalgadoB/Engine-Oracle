# Imports; 0% tariff
import torch
from torch import nn

# Decided to create different for clarity even if they share the same design

class classifier_year(nn.Module):
    """Neural Network Classifier for years."""
    def __init__(self, input_size: int, output_features: int) -> None:
        """Initializer method for the Neural Network. 
        Defines layers through nn.Sequential

        Args:
            input_size (int): Number of features that the model takes in.
            output_features (int): the number of classes/labels it can output.
        """
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_features)
        )

    def forward(self, x):
        return self.linear_stack(x)
    
    
class classifier_driver(nn.Module):
    """Neural Network Classifier for drivers."""
    def __init__(self, input_size: int, output_features: int) -> None:
        """Initializer method for the Neural Network. 
        Defines layers through nn.Sequential

        Args:
            input_size (int): Number of features that the model takes in.
            output_features (int): the number of classes/labels it can output.
        """
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_features)
        )

    def forward(self, x):
        return self.linear_stack(x)
    
    
class classifier_circuit(nn.Module):
    """Neural Network Classifier for circuits."""
    def __init__(self, input_size: int, output_features: int) -> None:
        """Initializer method for the Neural Network. 
        Defines layers through nn.Sequential

        Args:
            input_size (int): Number of features that the model takes in.
            output_features (int): the number of classes/labels it can output.
        """
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_features)
        )

    def forward(self, x):
        return self.linear_stack(x)