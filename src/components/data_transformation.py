# Data Transformation
# I have already saved the training, testing and raw data in artifacts
# I need to transform the data --> create a function data_transform that transforms the training and testing data
# Here we can use the initiate_data_ingestion function from DataIngestion class that returns the file path
# Then use the filepath for training and testing to transform it, I need to transform the independent vairables only and return the X_train and X_test after transformation
# Should I create a Data Transformation Class or not ?

import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# For Input things required for Data Transformation
@dataclass
class DataTransformationConfig:
    # Any paths or any inputs I will be requiring for data transformation
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        ## This is created to get my pickle files that will be used if I want to perform encoding, scaler etc
        try:
            # I already performed EDA and got the information on the numerical and categorical features I have
            numerical_feature=['reading_score', 'writing_score']
            categorical_features=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            # There are some missing values then we need to handle those missing values 
            # So we will be using pipelines
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoding",OneHotEncoder(drop="first",sparse_output=False)),
                    ("StandardScaling",StandardScaler())
                ]
            )
            logging.info(f"Numerical Columns: {numerical_feature}")
            logging.info(f"Categorical Columns: {categorical_features}")

            # To combine both the above pipeline together, we will do column transformer
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_feature),
                    ("cat_pipeline",cat_pipeline,categorical_features)
                ],
                remainder="passthrough",
                n_jobs=-1
            )
            
            return preprocessor

            
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read Train and Test Data Completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformation_object()
            target_column_name="math_score"
            numerical_feature=['reading_score', 'writing_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]
            
            logging.info("Applying preprocessing object to the training and testing dataframes")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info(f"Saved preprocessing object")
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
