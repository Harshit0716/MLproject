import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
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
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return a dictionary of test scores.
    """
    try:
        report = {}

        for model_name, model_instance in models.items():  # Iterate directly over the dictionary
            model_instance.fit(X_train, y_train)  # Fit the model

            y_train_pred = model_instance.predict(X_train)

            y_test_pred = model_instance.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    