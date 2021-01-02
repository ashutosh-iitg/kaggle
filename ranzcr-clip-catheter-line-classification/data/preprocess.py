import os
import copy
import glob
import numpy as np
import pandas as pd

class DataPreprocess():
    def __init__(self, config_dict):
        self.df = self.csv_to_dataframe(**config_dict)
        self.target_cols = self.df.columns[1:-2].tolist()
        self.num_targets = len(self.target_cols)

    def csv_to_dataframe(self, image_dir, csv_path):
        df = pd.read_csv(csv_path)
        df['image_path'] = [os.path.join(image_dir, (img + '.jpg')) for img in df['StudyInstanceUID']]
        return df