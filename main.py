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
    
    X = np.array(data["train"]["mfcc"])
    y = np.array(data["train"]["year_labels"])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=11
    )
    shape = X_train.shape
    input_size = shape[-2]*shape[-1]
    
    X_train = torch.from_numpy(X_train).type(torch.float).to(device)
    X_val = torch.from_numpy(X_val).type(torch.float).to(device)
    y_train = torch.from_numpy(y_train).type(torch.long).to(device)
    y_val = torch.from_numpy(y_val).type(torch.long).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = 16
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, shuffle=False
    )
    
    num_seasons = 2024 - 2017 + 1
    clf = classifier_year(input_size, num_seasons).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=clf.parameters(), lr=0.01)
    
    clf = train_classifier_dataloader(
        model=clf,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=500,
        loss_fn=loss_fn,
        optimizer=optimizer, 
        verbose=50,
        early_stop=25,
        device=device
    )
    
    X_test = np.array(data["test"]["mfcc"])
    y_test = np.array(data["test"]["year_labels"])
    
    X_test = torch.from_numpy(X_test).type(torch.float).to(device)
    y_test = torch.from_numpy(y_test).type(torch.long).to(device)
    
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
    
    X = np.array(data["train"]["mfcc"])
    y = np.array(data["train"]["driver_labels"])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=11
    )
    shape = X_train.shape
    input_size = shape[-2]*shape[-1]
    
    X_train = torch.from_numpy(X_train).type(torch.float).to(device)
    X_val = torch.from_numpy(X_val).type(torch.float).to(device)
    y_train = torch.from_numpy(y_train).type(torch.long).to(device)
    y_val = torch.from_numpy(y_val).type(torch.long).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = 16
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    drivers = get_drivers()
    clf = classifier_driver(input_size, len(drivers)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=clf.parameters(), lr=0.01)
    
    clf = train_classifier_dataloader(
        model=clf,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=200,
        loss_fn=loss_fn,
        optimizer=optimizer, 
        verbose=20,
        early_stop=20,
        device=device
    )
    
    X_test = np.array(data["test"]["mfcc"])
    y_test = np.array(data["test"]["driver_labels"])
    
    X_test = torch.from_numpy(X_test).type(torch.float).to(device)
    y_test = torch.from_numpy(y_test).type(torch.long).to(device)
    
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
    
    X = np.array(data["train"]["mfcc"])
    y = np.array(data["train"]["circuit_labels"])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=11
    )
    shape = X_train.shape
    input_size = shape[-2]*shape[-1]
    
    X_train = torch.from_numpy(X_train).type(torch.float).to(device)
    X_val = torch.from_numpy(X_val).type(torch.float).to(device)
    y_train = torch.from_numpy(y_train).type(torch.long).to(device)
    y_val = torch.from_numpy(y_val).type(torch.long).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = 16
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    circuits = get_circuits()
    clf = classifier_circuit(input_size, len(circuits)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=clf.parameters(), lr=0.01)
    
    clf = train_classifier_dataloader(
        model=clf,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=200,
        loss_fn=loss_fn,
        optimizer=optimizer, 
        verbose=20,
        early_stop=20,
        device=device
    )
    
    X_test = np.array(data["test"]["mfcc"])
    y_test = np.array(data["test"]["circuit_labels"])
    
    X_test = torch.from_numpy(X_test).type(torch.float).to(device)
    y_test = torch.from_numpy(y_test).type(torch.long).to(device)
    
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
    
    classify_years(data)
    
    
if __name__ == "__main__":
    main()