# Imports
import os

import numpy as np
import matplotlib.pyplot as plt
import librosa


def main():
    # Waveplot
    sampling_rate = 22050
    fig, ax = plt.subplots()
    signal, sr = librosa.load(
        "Data/2017/2017 Australian Grand Prix: Lewis Hamilton Onboard Pole Lap.m4a",
        sr=sampling_rate
    )
    librosa.display.waveshow(signal, sr=sr, color='silver')
    ax.set_title("Australia 2017 Pole")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time")
    plt.show()
    
    # Fast Fourier Transform
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))
    left_frequency = frequency[:len(frequency)//2]
    left_magnitude = magnitude[:len(magnitude)//2]
    
    fig, ax = plt.subplots()
    ax.plot(left_frequency, left_magnitude, color='silver')
    ax.set_title("Australia 2017 Pole FFT")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    plt.show()
    
    # Short Time Fourier Transform
    n_fft = 2048
    hop_length = 512
    fig, ax = plt.subplots()
    stft = librosa.core.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length
    )
    spectogram = np.abs(stft)
    log_spectogram = librosa.amplitude_to_db(spectogram)
    plot = librosa.display.specshow(
        log_spectogram,
        sr=sr,
        hop_length=hop_length,
        ax=ax
    )
    ax.set_title("Australia 2017 Pole Spectogram")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Time")
    plt.colorbar(plot)
    plt.show()
    
    # Mel Frequency Cepstral Coefficients
    fig, ax = plt.subplots()
    MFCC = librosa.feature.mfcc(
        y=signal,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=13
    )
    plot = librosa.display.specshow(
        MFCC,
        sr=sr,
        hop_length=hop_length,
        ax=ax
    )
    ax.set_title("Australia 2017 Pole MFCC")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Time")
    plt.colorbar(plot)
    plt.show()
    
    
if __name__ == "__main__":
    main()