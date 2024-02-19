import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    
    obj1 = DataTransformation()
    train_arr,test_arr,obj_path = obj1.initiate_transformation(train_data_path,test_data_path)

