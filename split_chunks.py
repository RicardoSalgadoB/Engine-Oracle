import os

from pydub import AudioSegment
from pydub.utils import make_chunks


def main():
    # Iterate through yearly data
    for year in range(2017, 2025):
        audio_files = os.listdir(f"Data/{year}")
        # Iterate through the audio files in each year directory
        for i, file in enumerate(audio_files):
            if file != "chunks":
                print(f"Splitting Video {i+1}/{len(audio_files)} in {year}")    # Give Feedback
                audio = AudioSegment.from_file(f"Data/{year}/{file}", "m4a")    # Get audio
                audio_len_ms = len(audio)
                # Get the engine part  of the audio
                audio = audio[int(audio_len_ms*0.1) : int(audio_len_ms*0.75)]   
                chunk_len_ms = 1000
                chunks = make_chunks(audio, chunk_len_ms)   # Split into 1 second chunks
                # Save each chunk
                for i, chunk in enumerate(chunks):
                    padding = 3 - len(str(i))
                    number = padding*"0" + str(i)
                    chunk.export(f"Data/{year}/chunks/{file}_{number}.wav", format="wav")
            

if __name__ == "__main__":
    main()