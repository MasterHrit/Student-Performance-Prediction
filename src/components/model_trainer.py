import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model

## We need to train all the models and get the result
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_data,test_data,preprocessor_path):
        
        try:
            # Now I will split the data to X and y
            logging.info("Train Data Split Started to X and y")
            X_train=train_data[:,:-1]
            y_train=train_data[:,-1]
            logging.info("Test Data Split Started to X and y")
            X_test=test_data[:,:-1]
            y_test=test_data[:,-1]
            # I could have also done below for splitting
            # X_train,y_train,X_test,y_test=(
            #     train_data[:,:-1],
            #     train_data[:,-1],
            #     test_data[:,:-1],
            #     test_data[:,-1]
            # )
            # Now I will train the model Linear Regression Model and get the accuracy in terms of r2 score
            models={
                "Linear Regression":LinearRegression(n_jobs=-1),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "SVR":SVR(),
                "KNNRegressor":KNeighborsRegressor(n_jobs=-1),
                "Decision_Tree_Regressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(n_jobs=-1),
                "AdaboostRegressor":AdaBoostRegressor(),
                "GradientBoostRegressor":GradientBoostingRegressor(),
                "XgBoostRegressor":XGBRegressor()
            }

            # One more way to get the parameter is create parameter file separately and read it here
            parameters={
                "Ridge":{
                    "alpha":[0.001, 0.01, 0.1, 1, 10, 100],
                    # "max_iter":[100,200,500,1000,2000,5000],
                    # "solver":["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
                },
                "Lasso":{
                    "alpha":[0.001, 0.01, 0.1, 1, 10, 100],
                    # "max_iter":[100,200,500,1000,2000,5000],
                    # "selection":["cyclic","random"]
                },
                "SVR":{
                    "kernel":["linear","poly","rbf","sigmoid"],
                    # "degree":[1,2,3,4,5,6],
                    # "gamma":["scale","auto"],
                    # "max_iter":[100,200,500,1000,2000,5000],
                    # "epsilon":[0.01, 0.05, 0.1, 0.2, 0.5]
                },
                "KNNRegressor":{
                    "n_neighbors":[i for i in range(1,11,3)]
                },
                "Decision_Tree_Regressor":{
                    "criterion":["squared_error","absolute_error","poisson"],
                    # "splitter":["best","random"],
                    # "max_depth":[1,2,3,4,5,6,7,8,9,10,15,20,25],
                    # "max_features":["auto","sqrt","log2"]
                },
                "RandomForestRegressor":{
                    "n_estimators":[100,200,500,1000],
                    # "criterion":["squared_error", "absolute_error", "poisson"],
                    # "max_depth":[5,8,10,15,"None"],
                    # "min_samples_split":[2,8,15,20],
                    # "max_features":[5,7,"auto",8]
                },
                "AdaboostRegressor":{
                    "n_estimators":[50,60,70,80,90,100],
                    # "loss":["linear", "square", "exponential"]
                },
                "GradientBoostRegressor":{
                    "loss":["squared_error","huber","absolute_error","quantile"],
                    # "criterion":["friedman_mse","squared_error"],
                    # "min_samples_split":[2,8,15,20],
                    # "n_estimators":[100,200,500,1000],
                    # "max_depth":[5,8,15,None,10]
                    #"learning_rate":[0.1,0.01,0.02,0.03]
                },
                "XgBoostRegressor":{
                    "n_estimators":[100,200,300],
                    # "learning_rate":[0.1,0.01],
                    # "max_depth":[5,8,12,20,30],
                    # "colsample_bytree":[0.5,0.8,1,0.3,0.4]
                }
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=parameters)
            
            ## To get the best model score from report
            best_model_score=max(model_report.values())
            ## To get best model name from dict
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            ## best_model_name, best_model_score = max(model_report.items(),key=lambda item: item[1])   -- Also an approach to find the max based on 2nd item i.e the score
            best_model=models[best_model_name]

            if best_model_score<0.6:
                logging.error("No Best Model Found")
                raise CustomException("No Best Model Found",sys)
            
            logging.info("Best Model Found")

            # Saving the model in pkl
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best Model Objected Saved")
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)            

            return (
                self.model_trainer_config.trained_model_file_path,
                best_model_name,
                r2_square
            )

        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
