import os
from re import A
import shutil

import moviepy.editor as mp
import moviepy.video.io.ffmpeg_tools as ffmpeg_tools
import pandas as pd


EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

DATAPATH = "data/MELD.Raw/"

def remove_unwanted_cols(dataframe):
    dataframe = dataframe.drop(columns=["Speaker", "StartTime", "EndTime", "Season", "Episode", "Sr No."])
    return dataframe

def extract_audio(data, datatype):

    if os.path.exists(os.path.join(DATAPATH, datatype + "_audio")):
        shutil.rmtree(os.path.join(DATAPATH, datatype + "_audio"))
    
    os.mkdir(os.path.join(DATAPATH, datatype + "_audio"))

    audio_csv_data = {"filepath": [], "label": [], "text": [], "sentiment": []}

    for idx, row in data.iterrows():
        dialogue_id = row["Dialogue_ID"]
        utterance_id = row["Utterance_ID"]
        video_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        video_path = os.path.join(DATAPATH, datatype, video_filename)
        
        audio_path = os.path.join(DATAPATH, datatype + "_audio", video_filename.replace(".mp4", ".mp3"))

        try:
            ffmpeg_tools.ffmpeg_extract_audio(video_path, audio_path, fps=16000)

            audio_csv_data["filepath"].append(audio_path)
            audio_csv_data["label"].append(row["Emotion"])
            audio_csv_data["text"].append(row["Utterance"])
            audio_csv_data["sentiment"].append(row["Sentiment"])

        except Exception as e:
            print("Exception:", e)
            print("Failed processing video:", video_path)


    return audio_csv_data

def get_dataset_information(datatype):
    data = pd.read_csv(os.path.join(DATAPATH, datatype + ".csv"))
    data = remove_unwanted_cols(data)

    return data

def create_dataset(datatype):
    data = get_dataset_information(datatype)

    audio_data_dict = extract_audio(data, datatype)

    audio_data = pd.DataFrame.from_dict(audio_data_dict)

    audio_data.to_csv(f"{datatype}_audio.csv")

create_dataset("test")
create_dataset("train")
create_dataset("val")