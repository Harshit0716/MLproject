
import sys
from dataclasses import dataclass 
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # Basically it is used to create a pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging
import os


@dataclass
class DataTransformationConfig: # it will give any path or inputs required for data transformation component
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl") # it is a path to store any model in pickle 
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):  # this func will help to create all pickle files which will be responsible in converting categorical features to numerical or to perform standard scaler and all.
        # This function is responsible for Data Transformation based on different types of data.
        try:
            numerical_columns=["reading_score","writing_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            #Creating a Pipline which is doing two important things 1) Handling missing values using Simpleimputer (median) 2) Doing Standard Scaling
            #This pipelines runs on the training data and then it will be used to transform the test data
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')), #for handling outliers using median strategy
                    ('std_scaler',StandardScaler())
                ]
            )
            # We are creating Cat_pipline which Handles missing values in categorical features
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')), #for handling missing values in categorical features
                    ('onehot',OneHotEncoder()), #for converting categorical features to numerical
                    ("scaler",StandardScaler(with_mean=False)) #for standard scaling
                ]

            )
            logging.info(f"Numerical columns Standard Scaling completed:{numerical_columns}")
            logging.info(f"Categorical columns encoding completed:{categorical_columns}")
            
            # Now we have to combine both the pipelines using ColumnTransformer
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)

                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(error_message="Data Transformation Failed",error_detail=sys)

    # Now lets start DataTransformation
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtainiiing the preprocessor object")


            preprocessing_object=self.get_data_transformer_object() 

            target_column_name="math_score"
            numerical_columns=["reading_score","writing_score"]

            #Train Data
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_train_df=train_df[target_column_name]

            #Test Data
            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_test_df=test_df[target_column_name]

            logging.info(
                f"Applying the preprocessor object on the training dataframe and testing dataframe"
            )
            input_feature_train_arr=preprocessing_object.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_features_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_test_df)
            ]

            logging.info(f"Saved preprocessing object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(error_message="Data Transformation Failed",error_detail=sys)