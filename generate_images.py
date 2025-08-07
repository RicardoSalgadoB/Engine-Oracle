# Imports
import os

import numpy as np
import matplotlib.pyplot as plt
import librosa

from tqdm import tqdm


def main():
    plt.style.use("ggplot")
    for d in os.listdir("Data"):    # For each year directory
        if d in [str(y) for y in range(2017, 2025)]:
            print(f"\nBEGINNING YEAR {d}")
            for f in tqdm(os.listdir(f"Data/{d}")): # Iterate through its audio files
                if f == "chunks":   # Make sure it isn't the dir for processed chunks
                    continue
                
                name = f.rstrip(".m4a") # Remove the file type from its name
                
                # WAVEPLOTS
                sampling_rate = 22050   # define sampling rate
                signal, sr = librosa.load(  # get the signal
                    f"Data/{d}/{f}",
                    sr=sampling_rate
                )
                
                fig, ax = plt.subplots()    # create plot
                librosa.display.waveshow(signal, sr=sr, color='black')  # display the signal
                ax.set_title(f"{name} waveplot", size=10)   # set up title and labels
                ax.set_ylabel("Amplitude")
                ax.set_xlabel("Time")
                plt.savefig(    # save the plot
                    f"Images/waveplots/{name.replace(' ', '_')}_waveplot.png"
                )
    
                # FAST FOURIER TRANSFORM
                fft = np.fft.fft(signal)    # get fft of signal
                magnitude = np.abs(fft)     # transform signal from complex to real
                frequency = np.linspace(0, sr, len(magnitude))  # get an array for the freq
                # FFTs are symmetrical so the right half isn't relevant
                left_frequency = frequency[:len(frequency)//2]
                left_magnitude = magnitude[:len(magnitude)//2]

                fig, ax = plt.subplots()    # create plot
                ax.plot(left_frequency, left_magnitude, color='black')  # plot the fft
                ax.set_title(f"{name} FFT", size=10)    # set up title and labels
                ax.set_xlabel("Frequency")
                ax.set_ylabel("Magnitude")
                plt.savefig(f"Images/fft/{name.replace(' ', '_')}_fft.png") # save
    
                # SHORT TIME FOURIER TRANSFORM
                n_fft = 2048    # Set up arguements
                hop_length = 512
                stft = librosa.core.stft(   # Process signal with STFT
                    signal,
                    n_fft=n_fft,
                    hop_length=hop_length
                )
                spectogram = np.abs(stft)   # From complex to real
                log_spectogram = librosa.amplitude_to_db(spectogram)    # Get the stft in decibels
                fig, ax = plt.subplots()    # Create plot
                plot = librosa.display.specshow(    # plot spectogram
                    log_spectogram,
                    sr=sr,
                    hop_length=hop_length,
                    ax=ax
                )
                ax.set_title(f"{name} STFT Spectogram", size=10)    # Config title and labels
                ax.set_ylabel("Frequency")
                ax.set_xlabel("Time")
                plt.colorbar(plot)  # Set colorbar
                plt.savefig(f"Images/spectograms/stft/{name.replace(' ', '_')}_stft.png")   # save
    
                # MEL FREQUENCY CEPSTRAL COEFFICIENTS
                MFCC = librosa.feature.mfcc(    # process signal with mfcc
                    y=signal,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mfcc=13
                )
                fig, ax = plt.subplots()    # create plot
                plot = librosa.display.specshow(
                    MFCC,
                    sr=sr,
                    hop_length=hop_length,
                    ax=ax
                )
                ax.set_title(f"{name} MFCC Spectogram", size=10)    # title and labels
                ax.set_ylabel("Frequency")
                ax.set_xlabel("Time")
                plt.colorbar(plot)
                plt.savefig(f"Images/spectograms/mfcc/{name}_mfcc.png")
    
    
if __name__ == "__main__":
    main()