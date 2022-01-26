import pandas as pd
import numpy as np
import moviepy.editor as mp

import os

DATAPATH = "data"

csv = pd.read_csv(os.path.join(DATAPATH, "dev_sent_emo.csv"))

print(csv.head())
