import os
from src.Chicken_Diesease_Classification import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.Chicken_Diesease_Classification.entity.config_entity import (DataTransformationConfig)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split(self, test_size = 0.25):
        print(self.config.data_path)
        data = pd.read_csv(self.config.data_path)
        train, test = train_test_split(data, test_size=test_size)
        print(train.head())
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
        print(train.shape)
        print(test.shape)
