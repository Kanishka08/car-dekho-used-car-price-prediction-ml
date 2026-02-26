import sys
import mlflow
import mlflow.sklearn
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from carprice.logger import logger
from carprice.exception import CustomException
from carprice.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    model_path: str = "artifacts/model/model.pkl"


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logger.info("Starting model training")

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(
                    n_estimators=100,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1
                )
            }

            best_rmse = float("inf")
            best_model = None

            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("Car-Price-Prediction")

            for model_name, model in models.items():
                with mlflow.start_run(run_name=model_name):
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    mae, rmse, r2 = evaluate_model(y_test, y_pred)

                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("mae", mae)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)

                    mlflow.sklearn.log_model(model, "model")

                    logger.info(
                        f"{model_name} -> RMSE: {rmse}, MAE: {mae}, R2: {r2}"
                    )

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model

            save_object(
                file_path=self.config.model_path,
                obj=best_model
            )

            logger.info("Best model saved successfully")
            return best_model

        except Exception as e:
            raise CustomException(e, sys)