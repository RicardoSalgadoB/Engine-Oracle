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
    random.seed(11)
    torch.manual_seed(11)
    np.random.seed(11)
    
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(11)
    elif torch.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(11)
    else:
        device = "cpu"
    
    batch_size = 16
    input_size = 572
    
    num_seasons = 2024 - 2017 + 1
    clf = classifier_year(input_size, num_seasons).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=clf.parameters(), lr=0.01)
        
    train_n = {1, 2, 3, 4, 5}
    test_n = 6
    
    X_test = []
    y_test = []
    for i, n in enumerate(list(train_n)):
        print(f"\nSUBSET {i+1}/{len(train_n)}")
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        for i in range(len(data["subsets"])):
            if data["subsets"][i] in list(train_n - {n}):
                X_train.append(data["mfcc"][i])
                y_train.append(data["year_labels"][i])
            if data["subsets"][i] == n:
                X_val.append(data["mfcc"][i])
                y_val.append(data["year_labels"][i])
                
        X_train = torch.tensor(X_train).type(torch.float).to(device)
        y_train = torch.tensor(y_train).type(torch.long).to(device)
        X_val = torch.tensor(X_val).type(torch.float).to(device)
        y_val = torch.tensor(y_val).type(torch.long).to(device)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
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
        
    for i in range(len(data["subsets"])):
        if data["subsets"][i] == test_n:
            X_test.append(data["mfcc"][i])
            y_test.append(data["year_labels"][i])
    
    X_test = torch.tensor(X_test).type(torch.float).to(device)
    y_test = torch.tensor(y_test).type(torch.long).to(device)
    
    test_classifier(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        classes=[y for y in range(2017, 2025)],
        loss_fn=loss_fn
    )
    

def classify_drivers(data: dict):
    random.seed(11)
    torch.manual_seed(11)
    np.random.seed(11)
    
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(11)
    elif torch.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(11)
    else:
        device = "cpu"
    
    batch_size = 16
    input_size = 572
    
    drivers = get_drivers()
    clf = classifier_driver(input_size, len(drivers)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=clf.parameters(), lr=0.01)
        
    train_n = {1, 2, 3, 4, 5}
    test_n = 6
    
    X_test = []
    y_test = []
    for i, n in enumerate(list(train_n)):
        print(f"\nSUBSET {i+1}/{len(train_n)}")
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        for i in range(len(data["subsets"])):
            if data["subsets"][i] in list(train_n - {n}):
                X_train.append(data["mfcc"][i])
                y_train.append(data["driver_labels"][i])
            if data["subsets"][i] == n:
                X_val.append(data["mfcc"][i])
                y_val.append(data["driver_labels"][i])
                
        X_train = torch.tensor(X_train).type(torch.float).to(device)
        y_train = torch.tensor(y_train).type(torch.long).to(device)
        X_val = torch.tensor(X_val).type(torch.float).to(device)
        y_val = torch.tensor(y_val).type(torch.long).to(device)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
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
        
    for i in range(len(data["subsets"])):
        if data["subsets"][i] == test_n:
            X_test.append(data["mfcc"][i])
            y_test.append(data["driver_labels"][i])
    
    X_test = torch.tensor(X_test).type(torch.float).to(device)
    y_test = torch.tensor(y_test).type(torch.long).to(device)
    
    test_classifier(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        classes=drivers,
        loss_fn=loss_fn
    )

def classify_circuits(data: dict):
    random.seed(11)
    torch.manual_seed(11)
    np.random.seed(11)
    
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(11)
    elif torch.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(11)
    else:
        device = "cpu"
    
    batch_size = 16
    input_size = 572
    
    circuits = get_circuits()
    clf = classifier_circuit(input_size, len(circuits)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=clf.parameters(), lr=0.01)
        
    train_n = {1, 2, 3, 4, 5}
    test_n = 6
    
    X_test = []
    y_test = []
    for i, n in enumerate(list(train_n)):
        print(f"\nSUBSET {i+1}/{len(train_n)}")
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        for i in range(len(data["subsets"])):
            if data["subsets"][i] in list(train_n - {n}):
                X_train.append(data["mfcc"][i])
                y_train.append(data["circuit_labels"][i])
            if data["subsets"][i] == n:
                X_val.append(data["mfcc"][i])
                y_val.append(data["circuit_labels"][i])
                
        X_train = torch.tensor(X_train).type(torch.float).to(device)
        y_train = torch.tensor(y_train).type(torch.long).to(device)
        X_val = torch.tensor(X_val).type(torch.float).to(device)
        y_val = torch.tensor(y_val).type(torch.long).to(device)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
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
        
    for i in range(len(data["subsets"])):
        if data["subsets"][i] == test_n:
            X_test.append(data["mfcc"][i])
            y_test.append(data["circuit_labels"][i])
    
    X_test = torch.tensor(X_test).type(torch.float).to(device)
    y_test = torch.tensor(y_test).type(torch.long).to(device)
    
    test_classifier(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        classes=circuits,
        loss_fn=loss_fn
    )
    
    
def main():
    #data = process_data(True)
    with open("Data/processed_data.json", "r") as fp:
        data = json.load(fp)
    
    classify_drivers(data)
    
    
if __name__ == "__main__":
    main()