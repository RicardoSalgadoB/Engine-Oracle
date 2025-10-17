from typing import Optional
from timeit import default_timer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch import nn
from torchmetrics import Recall, Precision, F1Score
from tqdm import tqdm


class EarlyStopping:
    """Handles early stopping of the model to avoid overfitting."""
    def __init__(self, patience: int = 5, delta: float = 0) -> None:
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
        
    def check_early_stop(self, val_loss):
        # continue
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            
        # approaching a stop
        else:
            self.no_improvement_count += 1
            #stop
            if self.no_improvement_count >= self.patience:
                self.stop_training = True


def accuracy_fn(y_pred, y_true):
    """Return the accuraccy of a given set of predictions and test values."""
    correct = torch.eq(y_pred, y_true).sum().item()
    return correct/len(y_pred)


def train_classifier(
    model: nn.Module,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    y_train: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int, 
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer, 
    verbose: Optional[int] = None,
    early_stop: Optional[int] = None,
    device: str = "cpu",
    time: bool = True
) -> nn.Module:
    """Handles model training given train and validation datasets (aka. tensors).
    The number of epochs that the model is going to be trained are to be specified.
    As well as the loss function and optimizers to be used.
    Finally, the number of iterations before metrics (verbose) and early stopping
        can also be specified as well as the device in which to train the model
        and if the training is to be tiemed.
        
    The funciton returns the trained model
    """
    # Reproducibility
    torch.manual_seed(11)
    
    if time:
        train_start = default_timer()
    
    # Set up an instance of the early stopping clas with the given early stop
    if early_stop:
        early_stopping = EarlyStopping(
            patience=early_stop,
            delta=0.0
        )
    else:
        early_stopping = None
    
    # Begin the training validation process
    for e in tqdm(range(epochs)):
        # train
        model.train()
        
        y_logits = model(X_train)
        y_probs = torch.softmax(y_logits, dim=1)
        y_pred = torch.argmax(y_probs, dim=1).type(torch.float)
        
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # eval
        model.eval()
        
        with torch.inference_mode():
            val_logits = model(X_val)
        val_probs = torch.softmax(val_logits, dim=1)
        val_pred = torch.argmax(val_probs, dim=1).type(torch.float)
        val_loss = loss_fn(val_logits, y_val)
        val_acc = accuracy_fn(val_pred, y_val)
        
        if early_stopping:
            early_stopping.check_early_stop(val_loss)
        
            if early_stopping.stop_training:
                print(f"\nEarly stopping at epoch {e}")
                break
        
        # if specified, give metrics after a certain number of rounds
        if verbose and e%verbose == 0:
            print(f"Epoch: {e}")
            print(f"Loss: {loss:.6f} | Acc: {acc:.3f}")
            print(f"Test Loss: {val_loss:.6f} | Test Acc: {val_acc:.3f}\n")
    
    if time:
        train_end = default_timer()
        print(f"The classifier took {train_end-train_start:.3f} seconds to train on {device}")
            
    return model


def train_classifier_dataloader(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    epochs: int, 
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer, 
    verbose: Optional[int] = None,
    early_stop: Optional[int] = None,
    device: str = "cpu",
    time: bool = True
) -> nn.Module:
    """This fucntion trains a classifier model given train & valdiation dataloaders.

    Args:
        model (nn.Module): The classfier model
        train_dataloader (torch.utils.data.DataLoader): Train dataloader
        val_dataloader (torch.utils.data.DataLoader): Validation dataloader
        epochs (int): Numebr of epochs to train the model
        loss_fn (nn.Module): The loss function to be used in training
        optimizer (torch.optim.Optimizer): The optimiezer to be used in training
        verbose (Optional[int], optional): Number of epochs after which to display metrics. 
            Defaults to None.
        early_stop (Optional[int], optional): Early stop rounds. Defaults to None.
        device (str, optional): Device to train the mdoel on. Defaults to "cpu".
        time (bool, optional): Whether if to time the model training time. 
            Defaults to True.

    Returns:
        nn.Module: The trained model
    """
    # Reproducibility
    torch.manual_seed(11)
    
    if time:
        train_start = default_timer()
    
    # Define instance of early stopping class
    if early_stop:
        early_stopping = EarlyStopping(
            patience=early_stop,
            delta=0.0
        )
    else:
        early_stopping = None
    
    # Begin training/validation
    for e in tqdm(range(epochs)):
        # train
        model.train()
        
        train_loss, train_acc = 0, 0
        for X_train, y_train in train_dataloader:
            y_logits = model(X_train)
            y_probs = torch.softmax(y_logits, dim=1)
            y_pred = torch.argmax(y_probs, dim=1).type(torch.float)

            loss = loss_fn(y_logits, y_train)
            acc = accuracy_fn(y_pred, y_train)
            
            train_loss += loss
            train_acc += acc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        
        # eval
        model.eval()
        
        val_loss, val_acc = 0, 0
        with torch.inference_mode():
            for X_val, y_val in val_dataloader:
                val_logits = model(X_val)
                val_probs = torch.softmax(val_logits, dim=1)
                val_pred = torch.argmax(val_probs, dim=1).type(torch.float)
                loss = loss_fn(val_logits, y_val)
                acc = accuracy_fn(val_pred, y_val)
                
                val_loss += loss
                val_acc += acc
                
            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)
        
        if early_stopping:
            early_stopping.check_early_stop(val_loss)
        
            if early_stopping.stop_training:
                print(f"\nEarly stopping at epoch {e}")
                break
        
        if verbose and e%verbose == 0:  # Display metrics in the indicated rounds
            print(f"Epoch: {e}")
            print(f"Loss: {loss:.6f} | Acc: {acc:.3f}")
            print(f"Validation Loss: {val_loss:.6f} | Validation Acc: {val_acc:.3f}\n")
    
    if time:
        train_end = default_timer()
        print(f"The classifier took {train_end-train_start:.3f} seconds to train on {device}")
    
    return model

    
def test_classifier(
    clf: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    classes: list,
    type: str,
    loss_fn: nn.Module,
    save_cm: bool = False,
    save_clf: bool = False,
) -> None:
    """Train a classifier model given the features and labels of a test dataset.

    Args:
        clf (nn.Module): The classfier model
        X_test (torch.Tensor): Features of the test dataset
        y_test (torch.Tensor): Labels of the test dataset
        classes (list): List with the names of the labels. Need for confussion matrix
        type (str): What are we classyfing (mainly, for the name of the cm file)
        loss_fn (nn.Module): Loss function used during training.
        save_cm (bool, optional): Whether to save the confussion matrix plot. 
            Defaults to False.
        save_clf (bool, optional): Whether to save the model state dict or not. 
            Default to False.
    """
    clf.eval()
    y_logits = clf(X_test)
    y_probs = torch.softmax(y_logits, dim=1)
    y_pred = torch.argmax(y_probs, dim=1).type(torch.float)
    
    # Define metrics
    rec = Recall(
        task='multiclass', 
        average='macro', 
        num_classes=len(classes)
    ).to('mps')
    prec = Precision(
        task='multiclass', 
        average='macro', 
        num_classes=len(classes)
    ).to('mps')
    f1 = F1Score(   # Not related to Formula 1
        task='multiclass', 
        average='macro', 
        num_classes=len(classes)
    ).to('mps')
    
    # Get and print metrics
    final_loss = loss_fn(y_logits, y_test)
    final_acc = accuracy_fn(y_pred, y_test)
    final_rec = rec(y_pred, y_test)
    final_prec = prec(y_pred, y_test)
    final_f1 = f1(y_pred, y_test)
    
    print(f"\nFinal Loss: {final_loss:.6f}")
    print(f"Final Accuracy: {final_acc:.3f}")
    print(f"Final Recall: {final_rec:.3f}")
    print(f"Final Precision: {final_prec:.3f}")
    print(f"Final F1-Score: {final_f1:.3f}")
    
    # Create and display confussion matrix
    cm = confusion_matrix(y_test.cpu(), y_pred.cpu())
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=classes
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"{type} Test Confussion Matrix")
    plt.xticks(rotation=60)
    
    # Save the confussion matrix if so
    if save_cm:
        plt.savefig(f"Images/confussion_matrices/{type}.png")
        
    plt.show()

    # Save the model if variables says so
    if save_clf:
        torch.save(clf.state_dict(), f"Models/{type}_clf_state_dict.pth")