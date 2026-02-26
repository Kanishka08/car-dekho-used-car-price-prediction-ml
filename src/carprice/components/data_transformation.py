import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from carprice.logger import logger
from carprice.exception import CustomException
from carprice.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = "artifacts/model/preprocessor.pkl"


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_preprocessor(self):
        try:
            numerical_cols = [
                "vehicle_age", "km_driven", "mileage",
                "engine", "max_power", "seats"
            ]

            categorical_cols = [
                "brand", "seller_type",
                "fuel_type", "transmission_type"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_path):
        logger.info("Starting data transformation")

        try:
            df = pd.read_csv(raw_data_path)

            # Drop duplicates
            df = df.drop_duplicates()

            # Drop high-cardinality columns
            df.drop(columns=["car_name", "model"], inplace=True)

            # Separate features and target
            X = df.drop("selling_price", axis=1)
            y = df["selling_price"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            preprocessor = self.get_preprocessor()

            # Fit on train, transform both
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Save preprocessor
            save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessor
            )

            logger.info("Data transformation completed successfully")

            return (
                X_train_processed,
                X_test_processed,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)