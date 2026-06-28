import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        logging.error(e)
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        for model_name,model in models.items():
            if model_name in param.keys():
                gs=GridSearchCV(model,param_grid=param[model_name],scoring="neg_mean_squared_error",n_jobs=-1,cv=3)
                gs.fit(X_train,y_train)
                y_train_pred=gs.predict(X_train)
                y_test_pred=gs.predict(X_test)
                models[model_name] = gs.best_estimator_ # Either you replace the model in the dictionary OR
                # model.set_params(**gs.best_params_) # Can also have set the parameter to the model in the dictionary and then trained and evaluated
            else:
                model.fit(X_train,y_train)
                y_train_pred=model.predict(X_train)
                y_test_pred=model.predict(X_test)
            

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[model_name]=test_model_score
            
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e,sys)