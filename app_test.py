import requests

def audio_file():
    url = "http://127.0.0.1:8000/predict/file"
    file_path = "Data/2023/Charles Leclerc's Pole Lap | 2023 Las Vegas Grand Prix | Pirelli.m4a"
    
    with open(file_path, 'rb') as f:
        files = {"audio_file": ("sample.m4a", f, "audio/m4a")}
        resp = requests.post(url=url, files=files)
    print(resp.json())
    
if __name__ == "__main__":
    audio_file()