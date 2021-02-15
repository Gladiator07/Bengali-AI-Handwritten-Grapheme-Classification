import pandas as pd
import numpy as np
import joblib
import glob
from tqdm import tqdm

if __name__=="__main__":
    files = glob.glob("/root/content/train_*.parquet")

    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.image_ids.values
        df = df.drop("image_id", axis=1)
        image_array = df.values
        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_array[j, :], f"../root/input/image_pickles/{img_id}.pkl")