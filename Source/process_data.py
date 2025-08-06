import os
import json

import numpy as np
import random

import librosa
from tqdm import tqdm


POLE_DRIVERS = {
    "Lewis Hamilton": 0,
    "Valtteri Bottas": 1,
    "Kimi Raikkonen": 2,
    "Sebastian Vettel": 3,
    "Daniel Ricciardo": 4,
    "Charles Leclerc": 5,
    "Max Verstappen": 6,
    "Lance Stroll": 7,
    "Lando Norris": 8,
    "Carlos Sainz": 9,
    "George Russell": 10,
    "Kevin Magnussen": 11,
    "Sergio Pérez": 12,
}

CIRUCUITS = {
    "Australian": 0,
    "Azerbaijan": 1,
    "Bahrain": 2,
    "Canadian": 3,
    "Chinese": 4,
    "Monaco": 5,
    "Spanish": 6,
    "Russian": 7,
    "US": 8,
    "United States": 8,
    "Japanese": 9,
    "Malaysian": 10,
    "Belgian": 11,
    "British": 12,
    "70th": 12,
    "Mexic": 13,    # Deliverate as it includes "Mexico" and "Mexican"
    "Hungarian": 14,
    "Singapore": 15,
    "Brazil": 16,   # Includes both "Brazil" and "Brazilian"
    "Sao Paulo": 17,
    "Austrian": 18,
    "Styrian": 18,
    "Abu Dhabi": 19,
    "Italian": 20,
    "French": 21,
    "German": 22,
    "Turkish": 23,
    "Portuguese": 24,
    "Tuscan": 25,
    "Eifel": 26,
    "Emilia Romagna": 27,
    "Sakhir": 28,
    "Qatar": 29,
    "Saudi": 30,
    "Dutch": 31,
    "Miami": 32,
    "Vegas": 33,
}

def process_data(save: bool = False) -> dict:
    random.seed(11)

    data = {
        "mfcc": [],
        "year_labels": [],
        "year": [],
        "driver_labels": [],
        "driver": [],
        "circuit_labels": [],
        "circuit": [],
        "subsets": [],
    }
    SAMPLE_RATE = 22050
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    expected_num_mfcc_vectors = np.ceil(SAMPLE_RATE / hop_length)
    
    data_path = "Data"
    total_samples = 0
    accepted_samples = 0
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):
        dirpath_components = dirpath.split("/")
        if dirpath_components[-1] == "chunks":
            year = dirpath_components[-2]
            print(f"\nStarting year {year}")
            
            for f in tqdm(filenames):
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(
                    file_path,
                    sr=SAMPLE_RATE
                )
    
                mfcc = librosa.feature.mfcc(
                    y=signal,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mfcc=n_mfcc
                )
                mfcc = mfcc.T.tolist()
                
                driver = ""
                driver_label = -1
                for d, label in POLE_DRIVERS.items():
                    if d in f:
                        driver = d
                        driver_label = label
                        break
                        
                circuit = ""
                circuit_label = -1
                for c, label in CIRUCUITS.items():
                    if c in f:
                        circuit = c
                        circuit_label = label
                        break

                if len(mfcc) == expected_num_mfcc_vectors:
                    total_samples += 1
                    if (driver != "" 
                        and driver_label >= 0 
                        and circuit != "" 
                        and circuit_label >= 0):
                        accepted_samples += 1
                        n = random.randint(1, 6)
                        data["mfcc"].append(mfcc)
                        data["year_labels"].append(int(year) - 2017)
                        data["year"].append(year)
                        data["driver_labels"].append(driver_label)
                        data["driver"].append(driver)
                        data["circuit_labels"].append(circuit_label)
                        data["circuit"].append(circuit)
                        data["subsets"].append(n)
                            
    print(f"{accepted_samples}/{total_samples} samples accepted")          
                        
    if save:
        with open(f"{data_path}/processed_data.json", "w") as fp:
            json.dump(data, fp, indent=4)
        
    return data
                        

def get_drivers():
    return list(POLE_DRIVERS.keys())
    
    
def get_circuits():
    s = set()
    circuit_names = []
    for circuit, label in CIRUCUITS.items():
        if label not in s:
            circuit_names.append(circuit)
            s.add(label)
            
    return circuit_names