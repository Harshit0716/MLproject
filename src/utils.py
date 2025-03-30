import os
import sys
import numpy as np
import pandas as pd
import dill

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