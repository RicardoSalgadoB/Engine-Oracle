import os

from pydub import AudioSegment
from pydub.utils import make_chunks


def main():
    for year in range(2017, 2025):
        audio_files = os.listdir(f"Data/{year}")
        for i, file in enumerate(audio_files):
            if file != "chunks":
                print(f"Splitting Video {i+1}/{len(audio_files)} in {year}")
                audio = AudioSegment.from_file(f"Data/{year}/{file}", "m4a")
                audio_len_ms = len(audio)
                audio = audio[int(audio_len_ms*0.1) : int(audio_len_ms*0.75)]
                chunk_len_ms = 1000
                chunks = make_chunks(audio, chunk_len_ms)
                for i, chunk in enumerate(chunks):
                    padding = 3 - len(str(i))
                    number = padding*"0" + str(i)
                    chunk.export(f"Data/{year}/chunks/{file}_{number}.wav", format="wav")
            

if __name__ == "__main__":
    main()