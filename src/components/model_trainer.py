# import os
# import sys
# from dataclasses import dataclass

# from catboost import CatBoostRegressor
# from sklearn.ensemble import (RandomForestRegressor , AdaBoostRegressor, GradientBoostingRegressor)
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, roc_auc_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
# from sklearn.model_selection import GridSearchCV
# import pickle
# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object, load_object
# from src.utils import evaluate_models


# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join('artifacts', 'model.pkl')
#     # model_report_file_path = os.path.join('artifacts', 'model_report.txt')
#     # best_model_file_path = os.path.join('artifacts', 'best_model.pkl')

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()


#     def initiate_model_trainer(self, train_array, test_array):  #preprocessor pathnot needed so i removed it
#         try:
#             logging.info("Splitting training and test data")
#             X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])  #All columns except the last (:-1) are input features (X).The last column (-1) is the output label (y).
#             models = { "Random Forest": RandomForestRegressor(), 
#                       "Decision Tree": DecisionTreeRegressor(), 
#                       "Gradient Boosting": GradientBoostingRegressor(),
#                       "Linear Regression": LinearRegression(),
#                       "XGBRegressor": XGBRegressor(),
#                       "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#                       "AdaBoost Regressor": AdaBoostRegressor(),
#             }
                      
#             model_report:dict = evaluate_models(X_train = X_train , y_train = y_train,X_test = X_test, y_test = y_test, models = models) #the evaulate function will be under utils.py

#             best_model_score = max(sorted(model_report.values()))
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score < 0.6:
#                 raise CustomException("No best model found")
            
#             logging.info(f"Best model found is {best_model_name} with score {best_model_score}")

#             #preprocessing_obj = load_object(preprocessor_path)  

#             save_object(file_path= self.model_trainer_config.trained_model_file_path, obj = best_model)  #this will create a model.pkl file in the artifacts folder

#             predicted = best_model.predict(X_test)
#             r2_square = r2_score(y_test, predicted)
#             logging.info(f"R2 score of the best model is {r2_square}")



#             pass
#         except Exception as e:
#             raise CustomException(e, sys)
        




import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {      #models dict contains different regression models
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {                                                                   #params is a dict contaiming keys as model names and values as snother dict containing hyperparameters for tuning
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256] #hyperparameter tuning  #this tells GridSearch to train RandomForestRegressor 6 times, each with a different number of trees.
                },
                "Gradient Boosting":{
                   
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},  #No parameters to tune here, it uses default settings.
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params) #param is for hyperparameter tuning
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)