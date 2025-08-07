import os
import json

import torch
from torch import nn

from Source.process_data import process_data, get_drivers, get_circuits
from Source.torch_utils import test_classifier
from Source.models.classifiers import (
    classifier_driver, 
    classifier_circuit, 
    classifier_year
)


def test_years(data: dict):
    """Test the years classifier if it has been saved.

    Args:
        data (dict): A dictionary containing the audio data.

    Raises:
        FileExistsError: If the years classifier has not been saved.
    """
    if not os.path.exists("Models/years_clf_state_dict.pth"):
        raise FileExistsError("Cannot test on a model that has not been saved")
    
    # Set up device
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(11)
    elif torch.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(11)
    else:
        device = "cpu"
    
    # Set up test subset
    test_n = 6
    
    X_test = []
    y_test = []
    # Get test data
    for i in range(len(data["subsets"])):
        if data["subsets"][i] == test_n:
            X_test.append(data["mfcc"][i])
            y_test.append(data["year_labels"][i])
    
    # Test data to tensors
    X_test = torch.tensor(X_test).type(torch.float).to(device)
    y_test = torch.tensor(y_test).type(torch.long).to(device)
   
    # Create instance of classifier 
    input_size = 572
    years = [y for y in range(2017, 2025)]
    clf = classifier_year(572, len(years)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Load the state dictionary into the classfier
    clf.load_state_dict(torch.load("Models/years_clf_state_dict.pth"))
    
    # Test the model
    test_classifier(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        classes=years,
        type="years",
        loss_fn=loss_fn,
        save_cm=False,
        save_clf=False,
    )
    

def test_drivers(data: dict):
    """Test the drivers classifier if it has been saved.

    Args:
        data (dict): A dictionary containing the audio data.

    Raises:
        FileExistsError: If the drivers classifier has not been saved.
    """
    if not os.path.exists("Models/drivers_clf_state_dict.pth"):
        raise FileExistsError("Cannot test on a model that has not been saved")
    
    # Set up device
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(11)
    elif torch.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(11)
    else:
        device = "cpu"
    
    # Set up test subset
    test_n = 6
    
    X_test = []
    y_test = []
    # Get test data
    for i in range(len(data["subsets"])):
        if data["subsets"][i] == test_n:
            X_test.append(data["mfcc"][i])
            y_test.append(data["driver_labels"][i])
    
    # Test data to tensors
    X_test = torch.tensor(X_test).type(torch.float).to(device)
    y_test = torch.tensor(y_test).type(torch.long).to(device)
   
    # Create instance of classifier 
    input_size = 572
    drivers = get_drivers()
    clf = classifier_driver(572, len(drivers)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Load the state dictionary into the classfier
    clf.load_state_dict(torch.load("Models/drivers_clf_state_dict.pth"))
    
    # Test the model
    test_classifier(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        classes=drivers,
        type="drivers",
        loss_fn=loss_fn,
        save_cm=False,
        save_clf=False,
    )
    

def test_circuits(data: dict):
    """Test the circuits classifier if it has been saved.

    Args:
        data (dict): A dictionary containing the audio data.

    Raises:
        FileExistsError: If the classifier has not been saved.
    """
    if not os.path.exists("Models/circuits_clf_state_dict.pth"):
        raise FileExistsError("Cannot test on a model that has not been saved")
    
    # Set up device
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(11)
    elif torch.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(11)
    else:
        device = "cpu"
    
    # Set up test subset
    test_n = 6
    
    X_test = []
    y_test = []
    # Get test data
    for i in range(len(data["subsets"])):
        if data["subsets"][i] == test_n:
            X_test.append(data["mfcc"][i])
            y_test.append(data["circuit_labels"][i])
    
    # Test data to tensors
    X_test = torch.tensor(X_test).type(torch.float).to(device)
    y_test = torch.tensor(y_test).type(torch.long).to(device)
   
    # Create instance of classifier 
    input_size = 572
    circuits = get_circuits()
    clf = classifier_circuit(572, len(circuits)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Load the state dictionary into the classfier
    clf.load_state_dict(torch.load("Models/circuits_clf_state_dict.pth"))
    
    # Test the model
    test_classifier(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        classes=circuits,
        type="circuits",
        loss_fn=loss_fn,
        save_cm=False,
        save_clf=False,
    )
    
    
def main():
    # Get the data dictionary
    process_data_path = "Data/processed_data.json"
    if os.path.exists(process_data_path):
        with open("Data/processed_data.json", "r") as fp:
            data = json.load(fp)
    else:
        data = process_data(True)
        
    classifier = "circuits"  # Specify which model to test
    if classifier.lower() == "years":
        test_years(data)
    elif classifier.lower() == "drivers":
        test_drivers(data)
    elif classifier.lower() == "circuits":
        test_circuits(data)


if __name__ == "__main__":
    main()