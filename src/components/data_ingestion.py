import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

# Initialize the data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts","train.csv") 
    test_data_path : str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","raw.csv")

# Create Data Ingestion Class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Starts")

        try:
            df = pd.read_csv(os.path.join("notebooks/data","train.csv"))
            logging.info("Dataset read as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Raw data has been created")

            train_data,test_data = train_test_split(df,test_size=0.3,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info("Exception occured at data ingestion stage")
            raise CustomException(e,sys)
        


        


