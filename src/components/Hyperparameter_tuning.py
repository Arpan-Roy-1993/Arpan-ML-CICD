import os
import sys
import numpy as np
import pandas as pd
import joblib

from dataclasses import dataclass
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting data into train and test sets.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 150],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                },
                "CatBoosting Regressor": {
                    'iterations': [100, 150],
                    'learning_rate': [0.05, 0.1],
                    'depth': [3, 6]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1]
                }
            }

            best_model = None
            best_model_name = None
            best_score = -np.inf

            for name, model in models.items():
                logging.info(f"Tuning model: {name}")
                if params.get(name):
                    grid_search = GridSearchCV(
                        model,
                        param_grid=params[name],
                        cv=5,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    tuned_model = grid_search.best_estimator_
                    logging.info(f"{name} best params: {grid_search.best_params_}")
                else:
                    tuned_model = model
                    tuned_model.fit(X_train, y_train)

                y_pred = tuned_model.predict(X_test)
                score = r2_score(y_test, y_pred)
                logging.info(f"{name} R2 Score: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = tuned_model
                    best_model_name = name

            logging.info(f"Best model: {best_model_name} with score: {best_score:.4f}")
            save_object(file_path=self.config.trained_model_file_path, obj=best_model)

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)
