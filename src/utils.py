import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import logging # Import logging for better error messages
# Assuming CustomException is defined elsewhere
# from your_module import CustomException

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
# def evaluate_models(X_train, y_train,X_test,y_test,models,param):
#     try:
#         report = {}
#
#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para=param[list(models.keys())[i]]
#
#             gs = GridSearchCV(model,para,cv=3)
#             gs.fit(X_train,y_train)
#
#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)
#
#             #model.fit(X_train, y_train)  # Train model
#
#             y_train_pred = model.predict(X_train)
#
#             y_test_pred = model.predict(X_test)
#
#             train_model_score = r2_score(y_train, y_train_pred)
#
#             test_model_score = r2_score(y_test, y_test_pred)
#
#             report[list(models.keys())[i]] = test_model_score
#
#         return report
#
#     except Exception as e:
#         raise CustomException(e, sys)



# Setup basic logging for MLflow (optional but recommended)
logging.basicConfig(level=logging.WARN, format="%(asctime)s %(levelname)s %(message)s")


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        # Set MLflow tracking URI if you're using a remote server
        # mlflow.set_tracking_uri("http://localhost:5000") # Example: for local MLflow UI
        mlflow.set_experiment("Model Evaluation Experiment") # Name your experiment

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            # Start an MLflow run for each model
            with mlflow.start_run(run_name=f"Model_{model_name}"):
                # Log model type
                mlflow.log_param("model_type", model_name)

                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)

                # Log   best parameters found by GridSearchCV
                mlflow.log_params(gs.best_params_)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                # Log metrics
                mlflow.log_metric("train_r2_score", train_model_score)
                mlflow.log_metric("test_r2_score", test_model_score)

                # Log the trained model
                mlflow.sklearn.log_model(model, "model")

                report[model_name] = test_model_score

        return report

    except Exception as e:
        # Assuming CustomException is defined to handle this
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
