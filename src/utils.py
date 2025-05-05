import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import logging
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException  # Move this to the top to avoid circular imports

def save_object(file_path,obj):
    from src.exception import CustomException

    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(error_message="Error in saving the object",error_detail=sys)
    """"
    With the help of this code in data transformation we are saving pickle in hard disk 
    """
def evaluate_model(X_train, y_train, X_test, y_test, models,param):
    """
    Evaluate multiple models and return a dictionary of test scores.
    """
    try:
        model_report = {}
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            if model_name in param and param[model_name]:  # Check if hyperparameters are provided
                logging.info(f"Performing GridSearchCV for {model_name}")
                grid_search = GridSearchCV(estimator=model, param_grid=param[model_name], cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            score = r2_score(y_test, y_pred)
            model_report[model_name] = score
            logging.info(f"{model_name} R² score: {score}")
        
        return model_report
    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {e}")
        raise e
    