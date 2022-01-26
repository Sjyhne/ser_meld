import pandas as pd

import random
import os

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

STATEMENT_MAP = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
}

TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.1
VAL_SPLIT = 0.2

def split_actors(datapath):
    train_actors, test_actors, val_actors = [], [], []

    actors = os.listdir(datapath)
    random.shuffle(actors)

    train_actors = actors[:int(len(actors) * TRAIN_SPLIT)]
    val_actors = actors[int(len(actors) * TRAIN_SPLIT) : int(len(actors) * (TRAIN_SPLIT + VAL_SPLIT))]
    test_actors = actors[-int(len(actors) * TEST_SPLIT):]

    return train_actors, val_actors, test_actors

def get_label(filepath):
    label = filepath.split("-")[2]
    return EMOTION_MAP[label]

def get_statement(filepath):
    statement = filepath.split("-")[4]
    return STATEMENT_MAP[statement]

def get_filepaths(datapath, actors):

    dataframe = pd.DataFrame(columns=["filepath", "label", "text"])

    for actor in actors:
        for filepath in os.listdir(os.path.join(datapath, actor)):
            filepath = os.path.join(datapath, actor, filepath)
            label = get_label(filepath)
            text = get_statement(filepath)
            datapoint = pd.DataFrame([[filepath, label, text]], columns=["filepath", "label", "text"])
            dataframe = pd.concat([dataframe, datapoint])

    return dataframe

def create_datasets():
    datapath = "data/speech"
    tra, vaa, tea = split_actors(datapath)

    train_data = get_filepaths(datapath, tra)
    val_data = get_filepaths(datapath, vaa)
    test_data = get_filepaths(datapath, tea)

    assert train_data["filepath"].duplicated().any() != True
    assert val_data["filepath"].duplicated().any() != True
    assert test_data["filepath"].duplicated().any() != True

    train_data.to_csv("train_data.csv")
    val_data.to_csv("val_data.csv")
    test_data.to_csv("test_data.csv")

create_datasets()