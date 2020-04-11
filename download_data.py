import sys, re, subprocess, youtube_dl, os
import numpy as np


def gettime(timestr):
    parts = timestr.split(":")
    if len(parts) == 2:
        return (int(parts[0]) * 60) + float(parts[1])
    if len(parts) == 1:
        return float(parts[0])
    raise ValueError("Only minutes:seconds supported")


def runcommand(commandarray):
    return subprocess.check_output(commandarray, stderr=subprocess.STDOUT).decode("utf-8")


options = {
    'format': 'bestaudio/best',
    'extractaudio': True,
    'audioformat': "wav",
    'outtmpl': 'data/temp/%(id)s.wav',
    'noplaylist': True,
    'nooverwrites': True,
}

vids = np.genfromtxt('data/video_list.csv', delimiter=',', dtype=np.str)
# print(vids)
for i in range(vids.shape[0]):
    print(i)
    outfile = ""
    start = gettime(vids[i, 1])
    duration = 15
    speaker_name = vids[i][-1]
    store_location = f'data/audio/{speaker_name}'
    os.makedirs(store_location, exist_ok=True)
    with youtube_dl.YoutubeDL(options) as ydl:
        video_id = ydl.extract_info(vids[i, 0], download=False).get("id", None)
        video_file = "data/temp/{}.wav".format(video_id)
        if not os.path.exists(video_file):
            ydl.download([vids[i, 0]])

    msala_samples = 0
    if speaker_name == 'Mohamed_Salah':
        msala_samples += 1
        outfile = f'{store_location}/{video_id}_{msala_samples}.wav'
        runcommand(["ffmpeg", "-ss", str(start), "-t", str(duration), "-y", "-i", video_file, outfile])

    else:
        num_sampels = 10
        for samp in range(10):
            outfile = f'{store_location}/{video_id}_{samp}.wav'
            print("cropping...")
            runcommand(["ffmpeg", "-ss", str(start), "-t", str(duration), "-y", "-i", video_file, outfile])
            start = (start + duration) + 10
print('Done.')
