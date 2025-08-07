import os
from statistics import mode

import numpy as np
import librosa
from tqdm import tqdm

import torch

from Source.process_data import (
    get_circuits,
    get_drivers,
)
from Source.models.classifiers import (
    classifier_circuit,
    classifier_driver,
    classifier_year,
)


def get_mfccs(file_name: str) -> list:
    """A function to get all the MFCCs of a given file.
    
    Args:
        file_name (str): The name of the file whose chunks are to be processed.

    Returns:
        list: A list with all the MFCCs.
    """
    # Define parameters
    SAMPLE_RATE = 22050
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    signal_len_s = 1    # Be sure to specify the length of the signal in minutes
    expected_num_mfcc_vectors = np.ceil(SAMPLE_RATE*signal_len_s / hop_length)
    
    # Define data path and the file name
    data_path = "Data"
    
    # Process all chunks that match the file name
    mfccs = []  # list to save
    for dirpath, dirnames, filenames in os.walk(data_path):
        dirpath_components = dirpath.split("/")
        if (dirpath_components[-1] == "chunks"):
            for f in filenames:
                if file_name == f[:-12]:
                    file_path = os.path.join(dirpath, f)
                    # Get signal
                    signal, sr = librosa.load(
                        file_path,
                        sr=SAMPLE_RATE
                    )

                    # Process signal with Mel Frequency Cepstral Coefficients
                    mfcc = librosa.feature.mfcc(
                        y=signal,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mfcc=n_mfcc
                    )
                    mfcc = mfcc.T.tolist()
                    
                    # If the signals is as expecte store it
                    if len(mfcc) == expected_num_mfcc_vectors:
                        mfccs.append(mfcc)
                        
    return mfccs
                        

def main():
    file_name = "Max Verstappen's Pole Lap | 2022 Emilia Romagna Grand Prix | Pirelli"
    mfccs = get_mfccs(file_name)
                        
    if not mfccs:
        raise FileExistsError(f"{file_name} not found. Remeber to remove '.m4a'")
                    
    # Create the classifier models
    input_size = 572    # 44 * 13 aka. len(mfcc) * n_mfcc
    years = [y for y in range(2017, 2025)]  # Get a list of years
    drivers = get_drivers()     # Get a list of drivers
    circuits = get_circuits()   # Get a list of circuits
    clf_years = classifier_year(input_size, len(years))
    clf_drivers = classifier_driver(input_size, len(drivers))
    clf_circuits = classifier_circuit(input_size, len(circuits))
    
    # Load the state dictionaries
    clf_years.load_state_dict(torch.load("Models/years_clf_state_dict.pth"))
    clf_drivers.load_state_dict(torch.load("Models/drivers_clf_state_dict.pth"))
    clf_circuits.load_state_dict(torch.load("Models/circuits_clf_state_dict.pth"))
    
    # set modesl to eval modes for better performance
    clf_years.eval()
    clf_drivers.eval()
    clf_circuits.eval()
    
    # Create lists to store the predicted labels
    year_preds = []
    driver_preds = []
    circuit_preds = []
    
    # Iterat through the stored MFCCs
    for mfcc in mfccs:
        # Pass the data from a list of lists to a tensor
        # Need to unflatten it or else the model internal flattening does not work
        X = torch.tensor(mfcc).unflatten(dim=0, sizes=(1, 44))
        
        # Get the logits for each category
        year_logits = clf_years(X)
        driver_logits = clf_drivers(X)
        circuit_logits = clf_circuits(X)
        
        # Get the probabilities for each category
        year_probs = torch.softmax(year_logits, dim=1)
        driver_probs = torch.softmax(driver_logits, dim=1)
        circuit_probs = torch.softmax(circuit_logits, dim=1)
        
        # Get the predicted labels for each category
        year_pred = torch.argmax(year_probs, dim=1)[0]
        driver_pred = torch.argmax(driver_probs, dim=1)[0]
        circuit_pred = torch.argmax(circuit_probs, dim=1)[0]
        
        # Add the predicted labels to each list
        year_preds.append(year_pred)
        driver_preds.append(driver_pred)
        circuit_preds.append(circuit_pred)
    
    # Get a plurality vote with the mode
    year_index = mode(year_preds)
    driver_index = mode(driver_preds)
    circuit_index = mode(circuit_preds)
    
    # Print the resulting predictions
    print(f"Predicted year: {years[year_index]}")
    print(f"Predicted driver: {drivers[driver_index]}")
    print(f"Predicted cirucit: {circuits[circuit_index]}")
    
    
if __name__ == "__main__":
    main()