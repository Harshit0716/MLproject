import os
import sys
from src.exception import CustomException   # For CustomException
from src.logger import logging           # For logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # This is used to create class variables

@dataclass     # It directly defines the class variables
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            logging.info("Entered the Data Ingestion Method or Component")
            data=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as pandas dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(data,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data Ingestion Completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(error_message="Data Ingestion Failed",error_detail=sys)
if __name__=="__main__":
    data_ingestion=DataIngestion()
    data_ingestion.initiate_data_ingestion()       