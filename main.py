# IMPORTS
import random
import json

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from Source.models.classifiers import (
    classifier_year, 
    classifier_driver,
    classifier_circuit,
)
from Source.torch_utils import (
    train_classifier,
    train_classifier_dataloader, 
    test_classifier
)
from Source.process_data import process_data, get_drivers, get_circuits


def classify_years(data: dict):
    """This funcition trains a year classifier and tests it,
    optionally saving its confussion matrix.

    Args:
        data (dict): A dictionary wiht the MFCC data and labels
    """
    # Reproducibility
    random.seed(11) # 11 in honor of the best second RB driver since 2018
    torch.manual_seed(11)
    np.random.seed(11)
    
    # Set up device for faster training, I hope
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(11)
    elif torch.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(11)
    else:
        device = "cpu"
    
    # Set batch size and input size for dataloader and the model
    batch_size = 16
    input_size = 572    # n_mfcc * len(mfcc), check processing data
    
    # Create classifier, loss function and optimizer
    num_seasons = 2024 - 2017 + 1   # Get number of seasons
    clf = classifier_year(input_size, num_seasons).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=clf.parameters(), lr=0.01)
    
    # Set up test and train subsets
    train_n = {1, 2, 3, 4, 5}
    test_n = 6
    
    X_test = []
    y_test = []
    # Iterate through the subsets in the train subsets
    for i, n in enumerate(list(train_n)):
        print(f"\nSUBSET {i+1}/{len(train_n)}")
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        # Get subsets into test and validation
        for i in range(len(data["subsets"])):
            if data["subsets"][i] in list(train_n - {n}):
                X_train.append(data["mfcc"][i])
                y_train.append(data["year_labels"][i])
            if data["subsets"][i] == n:
                X_val.append(data["mfcc"][i])
                y_val.append(data["year_labels"][i])
        
        # Convert train and test lists into tensors and convert them to device
        X_train = torch.tensor(X_train).type(torch.float).to(device)
        y_train = torch.tensor(y_train).type(torch.long).to(device)
        X_val = torch.tensor(X_val).type(torch.float).to(device)
        y_val = torch.tensor(y_val).type(torch.long).to(device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # 
        clf = train_classifier_dataloader(
            model=clf,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=100,
            loss_fn=loss_fn,
            optimizer=optimizer, 
            verbose=20,
            early_stop=25,
            device=device
        )
    
    # Get test data  
    for i in range(len(data["subsets"])):
        if data["subsets"][i] == test_n:
            X_test.append(data["mfcc"][i])
            y_test.append(data["year_labels"][i])
    
    # Convert test data to torch tensors
    X_test = torch.tensor(X_test).type(torch.float).to(device)
    y_test = torch.tensor(y_test).type(torch.long).to(device)
    
    # Test the classifier
    test_classifier(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        classes=[y for y in range(2017, 2025)],
        type="years",
        loss_fn=loss_fn,
    )
    

def classify_drivers(data: dict):
    """This funcition trains a driver classifier and tests it,
    optionally saving its confussion matrix.

    Args:
        data (dict): A dictionary wiht the MFCC data and labels
    """
    # Reproducibility
    random.seed(11) # 11 in honor of the best second RB driver since 2018
    torch.manual_seed(11)
    np.random.seed(11)
    
    # Set up device for faster training, I hope
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(11)
    elif torch.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(11)
    else:
        device = "cpu"
    
    # Set batch size and input size for dataloader and the model
    batch_size = 16
    input_size = 572    # n_mfcc * len(mfcc), check processing data
    
    # Create classifier, loss funciton and optimizers
    drivers = get_drivers()     # Get list of drivers
    clf = classifier_driver(input_size, len(drivers)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=clf.parameters(), lr=0.01)
    
    # Set up train and test datasets
    train_n = {1, 2, 3, 4, 5}
    test_n = 6
    
    X_test = []
    y_test = []
    # Iterate through train subsets
    for i, n in enumerate(list(train_n)):
        print(f"\nSUBSET {i+1}/{len(train_n)}")
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        # Set up lists with train and validation subsets
        for i in range(len(data["subsets"])):
            if data["subsets"][i] in list(train_n - {n}):
                X_train.append(data["mfcc"][i])
                y_train.append(data["driver_labels"][i])
            if data["subsets"][i] == n:
                X_val.append(data["mfcc"][i])
                y_val.append(data["driver_labels"][i])
        
        # Get train and validation to torch tensros  
        X_train = torch.tensor(X_train).type(torch.float).to(device)
        y_train = torch.tensor(y_train).type(torch.long).to(device)
        X_val = torch.tensor(X_val).type(torch.float).to(device)
        y_val = torch.tensor(y_val).type(torch.long).to(device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Train classifier
        clf = train_classifier_dataloader(
            model=clf,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=50,
            loss_fn=loss_fn,
            optimizer=optimizer, 
            verbose=25,
            early_stop=25,
            device=device
        )
    
    # Get test data
    for i in range(len(data["subsets"])):
        if data["subsets"][i] == test_n:
            X_test.append(data["mfcc"][i])
            y_test.append(data["driver_labels"][i])
    
    # Covnert test data to torch tensors
    X_test = torch.tensor(X_test).type(torch.float).to(device)
    y_test = torch.tensor(y_test).type(torch.long).to(device)
    
    # Test the classifier model
    test_classifier(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        classes=drivers,
        type="drivers",
        loss_fn=loss_fn
    )


def classify_circuits(data: dict):
    """This funcition trains a circuit classifier and tests it, 
    optionally saving teh resulting confussion matrix.

    Args:
        data (dict): A dictionary wiht the MFCC data and labels
    """
    # Reproducibility
    random.seed(11) # 11 in honor of the best second RB driver since 2018
    torch.manual_seed(11)
    np.random.seed(11)
    
    # Set up device for faster training, I hope
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(11)
    elif torch.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(11)
    else:
        device = "cpu"
    
    # Set batch size and input size for dataloader and the model
    batch_size = 16
    input_size = 572    # n_mfcc * len(mfcc), check processing data
    
    # Create classifier, loss function and optimizer
    circuits = get_circuits()   # Get a lsit of circuit names
    clf = classifier_circuit(input_size, len(circuits)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=clf.parameters(), lr=0.01)
        
    # Set up train and test subsets
    train_n = {1, 2, 3, 4, 5}
    test_n = 6
    
    X_test = []
    y_test = []
    # Iterate through train subsets
    for i, n in enumerate(list(train_n)):
        print(f"\nSUBSET {i+1}/{len(train_n)}")
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        # Get train and validdation data
        for i in range(len(data["subsets"])):
            if data["subsets"][i] in list(train_n - {n}):
                X_train.append(data["mfcc"][i])
                y_train.append(data["circuit_labels"][i])
            if data["subsets"][i] == n:
                X_val.append(data["mfcc"][i])
                y_val.append(data["circuit_labels"][i])
                
        # Convert train and validation data to torch tensors
        X_train = torch.tensor(X_train).type(torch.float).to(device)
        y_train = torch.tensor(y_train).type(torch.long).to(device)
        X_val = torch.tensor(X_val).type(torch.float).to(device)
        y_val = torch.tensor(y_val).type(torch.long).to(device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Train Classfier
        clf = train_classifier_dataloader(
            model=clf,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=50,
            loss_fn=loss_fn,
            optimizer=optimizer, 
            verbose=25,
            early_stop=25,
            device=device
        )
    
    # Get test data
    for i in range(len(data["subsets"])):
        if data["subsets"][i] == test_n:
            X_test.append(data["mfcc"][i])
            y_test.append(data["circuit_labels"][i])
    
    # Test data to tensors
    X_test = torch.tensor(X_test).type(torch.float).to(device)
    y_test = torch.tensor(y_test).type(torch.long).to(device)
    
    # Test the classifier
    test_classifier(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        classes=circuits,
        type="circuits",
        loss_fn=loss_fn
    )
    
    
def main():
    # Get the data dictionary
    #data = process_data(True)
    with open("Data/processed_data.json", "r") as fp:
        data = json.load(fp)
    
    # Classify stuff :)
    classify_drivers(data)
    
    
if __name__ == "__main__":
    main()