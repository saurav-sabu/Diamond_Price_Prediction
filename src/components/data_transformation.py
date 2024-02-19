from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np
import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        logging.info("Data Transformation Method Starts")

        try:
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_categories = ["Fair","Good","Very Good","Premium","Ideal"]
            color_categories = ["D","E","F","G","H","I","J"]
            clarity_categories = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]

            logging.info("Data Transformation Pipeline Initiated")

            num_pipeline = Pipeline(
                steps = [
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps= [
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("oe",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ("sc",StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_cols),
                ("cat_pipeline",cat_pipeline,categorical_cols)
            ])

            logging.info("Data Transformation completed")

            return preprocessor

        except Exception as e:
            logging.info("Exception occurred at data transformation stage")
            raise CustomException(e,sys)
        
    
    def initiate_transformation(self,train_data_path,test_data_path):

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read train and test data")
            logging.info(f"Train Data head: \n{train_df.head().to_string()}")
            logging.info(f"Test Data head: \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformation_obj()

            input_feature_train_df = train_df.drop(["id","price"],axis=1)
            target_feature_train_df = train_df["price"]

            input_feature_train_df = train_df.drop(["id","price"],axis=1)
            target_feature_train_df = train_df["price"]

            input_feature_test_df = test_df.drop(["id","price"],axis=1)
            target_feature_test_df = test_df["price"]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object")
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        


        