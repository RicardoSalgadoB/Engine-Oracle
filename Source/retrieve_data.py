from pytubefix import YouTube, Playlist
from pytubefix.cli import on_progress


def main():
    # Get a list of playlists containing pole laps between 2017 and 2024
    playlists = {
        "2017" : "https://www.youtube.com/watch?v=LqpkFMYYMxs&list=PLfoNZDHitwjXAOD6FUnDpwWDyuIbJZe__",
        "2018" : "https://www.youtube.com/watch?v=xyX6aNxL9SQ&list=PLfoNZDHitwjVNx_OW8yp2smxjbbOuCZGp",
        "2019" : "https://www.youtube.com/watch?v=kmuKQ2JQK30&list=PLfoNZDHitwjUA9aqbPGKw1l4SIz2bACi_",
        "2020" : "https://www.youtube.com/watch?v=qdf-7a4tPRk&list=PLfoNZDHitwjUG6Nq8W0XLC_ke3s90wb3M",
        "2021" : "https://www.youtube.com/watch?v=jwJOmeDjX8g&list=PLfoNZDHitwjWgczXBINGGl4mmkfas1_Pe",
        "2022" : "https://www.youtube.com/watch?v=jSIAT0UYotQ&list=PLCvaDWh6BegKHAs21Wg1wFcWsYn_081IB",
        "2023" : "https://www.youtube.com/watch?v=RMpWukELqCc&list=PLCvaDWh6BegKWKWT69oP0jxnHbLbWjgM7",
        "2024" : "https://www.youtube.com/watch?v=r7Mikgrm52k&list=PL1HouA-yvTAWxnczWCWl-CJ4CZOm2TNHP",
    }
    
    # The actual Leclerc Pole lap is already downloaded, don't need the 360
    bad_videos = [
        "360 CAM: Charles Leclerc Takes Pole Position | 2024 Monaco Grand Prix",
        "2017 Mexico Grand Prix: Best Of Team Radio"
    ]
    
    # Iterate through the playlist dictionary
    for year, url in playlists.items():
        p = Playlist(url)
        # Give some feedback to the user
        print(f"Starting downloading poles of year {year}")
        
        for i, video in enumerate(p.videos):
            # Again more feedback
            print(f"Video {i}/{len(p.videos)} in {year}")
            # If the video is good, download its audio
            if video.title not in bad_videos:
                ys = video.streams.get_audio_only()
                ys.download(output_path=f"Data/{year}")
    
    
if __name__ == "__main__":
    main()